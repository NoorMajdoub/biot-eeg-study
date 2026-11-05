"""
Per-(dataset, arm) Optuna hyperparameter search for BIOT fine-tuning.

Does NOT modify run_binary_supervised.py -- imports LitModel_finetune and
the per-dataset prepare_*_dataloader functions from it directly, and only
subclasses LitModel_finetune to add a linear-warmup + cosine-decay scheduler
(the original configure_optimizers returns [optimizer] only -- no scheduler
exists today). Gradient clipping needs no model-code change either: it's
passed straight into pl.Trainer(gradient_clip_val=...).

Scope: the 6 BIOT datasets that use run_binary_supervised.py (chbmit,
clinical, dream, karaone, yang2025, japaneeg). isruc and thoughtviz use
run_multiclass_supervised.py, which has its own LitModel_finetune class not
yet inspected -- they need the same treatment as a follow-up once we've
looked at that file.

Search space (see biot_scaling_job.sbatch's ORIGINAL_ROOT paths -- these
must match exactly):
  - lr: loguniform(1e-4, 5e-4) if arm=pretrained, loguniform(5e-4, 5e-3) if arm=random
  - grad_clip: loguniform(0.1, 5.0) -- searched independently per arm
  - warmup_ratio: uniform(0.0, 0.15) of total training steps (linear warmup,
    then cosine decay) -- searched independently per arm (deliberately NOT
    forced to match between pretrained/random, unlike batch_size below)
  - batch_size: FIXED per dataset, not searched -- must be identical between
    the pretrained and random arms of a dataset, and matches
    biot_scaling_job.sbatch's real production value exactly for each
    dataset (chbmit=512 [no override there -> argparse default], dream=16,
    clinical/karaone/yang2025/japaneeg=32). See DATASET_CONFIG.
  - weight_decay: fixed 1e-5 (both arms) -- not searched
  - optimizer: Adam (both arms) -- not searched
  - epochs: fixed at the existing default (100), governed in practice by the
    existing EarlyStopping(monitor="val_auroc", patience=5) -- not searched,
    not changed
  - seed: fixed at 42 for every trial in the search (this is a search-phase
    convention only -- the real scaling runs still use seeds 1/2/42)
  - data: always the 100% (full) split for this dataset -- the search never
    touches the 50pct/25pct scaling folders
  - precision: bf16 (compute-efficiency only, not part of the search
    space -- doesn't change what's being measured, just speeds up training)

Each (dataset, arm) pair is its own Optuna study of --n_trials trials
(default 20), pruned with MedianPruner watching val_auroc after every
epoch (matching the metric EarlyStopping already uses in production).

Usage (one study per invocation -- see submit_biot_optuna_job.sh for the
sbatch wrapper that loops over all 6 datasets x 2 arms):
  python biot_optuna_search.py --dataset chbmit --arm pretrained --n_trials 20
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

sys.path.insert(0, "/home/nourmaj/BIOT")

from run_binary_supervised import (  # noqa: E402
    LitModel_finetune,
    prepare_CHB_MIT_dataloader,
    prepare_Clinical_dataloader,
    prepare_Dream_dataloader,
    prepare_JapanEEG_dataloader,
    prepare_KaraOne_dataloader,
    prepare_Yang2025_dataloader,
    prepare_ThingsEEG1_dataloader,
)
from model import BIOTClassifier  # noqa: E402

PRETRAIN_CKPT = "pretrained-models/EEG-SHHS+PREST-18-channels.ckpt"

# dataset key -> (BIOT_NAME used by prepare_*_dataloader/args.dataset,
#                 loader function, in_channels, n_classes, sample_length,
#                 100%-data root, batch_size -- ALL must match
#                 biot_scaling_job.sbatch exactly. batch_size is FIXED here,
#                 not searched -- it must be identical between the pretrained
#                 and random arms of the same dataset, and matching production
#                 exactly matters (chbmit in particular uses no --batch_size
#                 override at all in biot_scaling_job.sbatch, so it runs on
#                 the script's own default of 512 -- NOT one of the smaller
#                 values the other datasets use).
DATASET_CONFIG = {
    "chbmit": dict(
        BIOT_NAME="CHB-MIT", loader=prepare_CHB_MIT_dataloader,
        in_channels=16, n_classes=1, sample_length=10,
        root="/scratch/nourmaj/chb-mit-processed-80h",
        batch_size=512, token_size=200, hop_length=100,  # biot_scaling_job.sbatch sets no override -> argparse default
    ),
    "clinical": dict(
        BIOT_NAME="Clinical", loader=prepare_Clinical_dataloader,
        in_channels=19, n_classes=1, sample_length=10,
        root="/scratch/nourmaj/ds004504_processed",
        batch_size=32, token_size=200, hop_length=100,
    ),
    "dream": dict(
        BIOT_NAME="Dream", loader=prepare_Dream_dataloader,
        in_channels=257, n_classes=1, sample_length=20,
        root="/scratch/nourmaj/dream_processed",
        batch_size=16, token_size=200, hop_length=100,
    ),
    "karaone": dict(
        BIOT_NAME="KaraOne", loader=prepare_KaraOne_dataloader,
        in_channels=60, n_classes=1, sample_length=5,
        root="/scratch/nourmaj/karaone_processed",
        batch_size=32, token_size=200, hop_length=100,
    ),
    "yang2025": dict(
        BIOT_NAME="Yang2025", loader=prepare_Yang2025_dataloader,
        in_channels=59, n_classes=1, sample_length=6,
        root="/scratch/nourmaj/yang2025_processed",
        batch_size=32, token_size=200, hop_length=100,
    ),
    "japaneeg": dict(
        BIOT_NAME="JapanEEG", loader=prepare_JapanEEG_dataloader,
        in_channels=128, n_classes=1, sample_length=10,
        # corrected ~80h-trimmed base -- same one used everywhere else now
        root="/scratch/nourmaj/japaneeg_processed_80h",
        batch_size=32, token_size=200, hop_length=100,
    ),
    "thingseeg1": dict(
        BIOT_NAME="ThingsEEG1", loader=prepare_ThingsEEG1_dataloader,
        in_channels=63, n_classes=1, sample_length=0.5,
        root="/scratch/nourmaj/things_eeg1/splits",
        batch_size=32, token_size=20, hop_length=10,
    ),
}

LR_RANGE = {
    "pretrained": (1e-4, 5e-4),
    "random": (5e-4, 5e-3),
}


class LitModel_finetune_optuna(LitModel_finetune):
    """Same as LitModel_finetune, only configure_optimizers is overridden
    to add a linear-warmup-then-cosine-decay scheduler, and
    validation_step/test_step are overridden ONLY to add a .float() cast
    before .cpu().numpy() -- under real bf16 precision (this Lightning
    version's "bf16" runs the whole model in bfloat16, not autocast-mixed),
    the original code's torch.sigmoid(prob).cpu().numpy() crashes because
    numpy can't hold a bfloat16 tensor directly. Everything else
    (training_step, validation_epoch_end, test_epoch_end, ...) is
    inherited unchanged from the real, already-verified class."""

    def __init__(self, args, model, warmup_ratio):
        super().__init__(args, model)
        self.warmup_ratio = warmup_ratio

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            prob = self.model(X)
            step_result = torch.sigmoid(prob).float().cpu().numpy()
            step_gt = y.cpu().numpy()
        return step_result, step_gt

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = torch.sigmoid(convScore).float().cpu().numpy()
            step_gt = y.cpu().numpy()
        return step_result, step_gt

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        total_steps = max(1, self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(self.warmup_ratio * total_steps))

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(warmup_steps)
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            progress = min(1.0, progress)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]


class OptunaTrackingCallback(Callback):
    """Reports val_auroc to Optuna after every epoch (for pruning) and
    separately tracks the BEST val_auroc seen across all epochs, so a
    trial's final score isn't just whatever epoch EarlyStopping happened
    to stop on (patience=5 means it stops a few epochs past the peak)."""

    def __init__(self, trial, monitor="val_auroc"):
        self.trial = trial
        self.monitor = monitor
        self.best_score = 0.0

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        current = float(current)
        if current > self.best_score:
            self.best_score = current
        self.trial.report(current, step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def build_args(dataset_key, arm, lr, seed=42):
    cfg = DATASET_CONFIG[dataset_key]
    args = argparse.Namespace()
    args.dataset = cfg["BIOT_NAME"]
    args.model = "BIOT"
    args.in_channels = cfg["in_channels"]
    args.n_classes = cfg["n_classes"]
    args.sample_length = cfg["sample_length"]
    args.sampling_rate = 200
    args.token_size = cfg["token_size"]
    args.hop_length = cfg["hop_length"]
    args.epochs = 100
    args.lr = lr
    args.weight_decay = 1e-5
    # Fixed, not searched -- must match production exactly and be identical
    # between the pretrained/random arms of the same dataset (see
    # DATASET_CONFIG's comment on why chbmit=512 while the rest are smaller).
    args.batch_size = cfg["batch_size"]
    args.num_workers = 8
    args.seed = seed
    args.pretrain_model_path = PRETRAIN_CKPT if arm == "pretrained" else ""
    return args


def objective(trial, dataset_key, arm, fraction):
    cfg = DATASET_CONFIG[dataset_key]

    lr = trial.suggest_float("lr", *LR_RANGE[arm], log=True)
    use_grad_clip = trial.suggest_categorical("use_grad_clip", [True, False])
    if use_grad_clip:
        grad_clip = trial.suggest_float("grad_clip", 0.1, 5.0, log=True)
    else:
        grad_clip = None
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.15)

    args = build_args(dataset_key, arm, lr, seed=42)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data root now depends on which fraction we're tuning for -- 100% was
    # already done in the first round; this run is for 50% or 25%.
    if fraction == "100":
        data_root = cfg["root"]
    else:
        data_root = f"{cfg['root']}_{fraction}pct"
    os.environ["BIOT_DATA_ROOT_OVERRIDE"] = data_root
    train_loader, test_loader, val_loader = cfg["loader"](args)

    model = BIOTClassifier(
        n_classes=args.n_classes,
        n_channels=args.in_channels,
        n_fft=args.token_size,
        hop_length=args.hop_length,
    )
    if args.pretrain_model_path and args.sampling_rate == 200:
        ckpt = torch.load(args.pretrain_model_path)
        model_state = model.biot.state_dict()
        skip_keys = [
            k for k in ckpt
            if k in model_state and ckpt[k].shape != model_state[k].shape
        ]
        filtered_ckpt = {k: v for k, v in ckpt.items() if k not in skip_keys}
        model.biot.load_state_dict(filtered_ckpt, strict=False)

    lightning_model = LitModel_finetune_optuna(args, model, warmup_ratio)

    early_stop_callback = EarlyStopping(
        monitor="val_auroc", patience=5, verbose=False, mode="max"
    )
    tracking_callback = OptunaTrackingCallback(trial, monitor="val_auroc")

    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        benchmark=True,
        enable_checkpointing=False,
        logger=False,
        max_epochs=args.epochs,
        gradient_clip_val=grad_clip,
        precision="bf16",
        callbacks=[early_stop_callback, tracking_callback],
    )

    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    return tracking_callback.best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG))
    parser.add_argument("--arm", required=True, choices=["pretrained", "random"])
    parser.add_argument("--fraction", default="100", choices=["100", "50", "25"])
    parser.add_argument("--n_trials", type=int, default=20)
    cli_args = parser.parse_args()

    out_dir = "/scratch/nourmaj/biot_optuna_results"
    os.makedirs(out_dir, exist_ok=True)
    # fraction is baked into the db/study/output names so the 50%/25% searches
    # never collide with (or overwrite) the already-completed 100% ones.
    tag = f"{cli_args.dataset}_{cli_args.arm}_{cli_args.fraction}pct"
    db_path = f"{out_dir}/{tag}.db"
    study_name = f"biot_{tag}"

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )

    n_done = len(
        [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    )
    n_remaining = max(0, cli_args.n_trials - n_done)
    print(f"=== BIOT Optuna search: dataset={cli_args.dataset} arm={cli_args.arm} fraction={cli_args.fraction} ===")
    print(f"{n_done} trials already recorded in {db_path}, running {n_remaining} more")

    if n_remaining > 0:
        study.optimize(
            lambda trial: objective(trial, cli_args.dataset, cli_args.arm, cli_args.fraction),
            n_trials=n_remaining,
        )

    print("=== BEST TRIAL ===")
    print(f"dataset={cli_args.dataset} arm={cli_args.arm} fraction={cli_args.fraction}")
    print(f"best val_auroc = {study.best_value}")
    print(f"best params = {study.best_params}")

    with open(f"{out_dir}/{tag}_best.json", "w") as f:
        json.dump(
            {
                "dataset": cli_args.dataset,
                "arm": cli_args.arm,
                "fraction": cli_args.fraction,
                "best_val_auroc": study.best_value,
                "best_params": study.best_params,
                "weight_decay": 1e-5,
                "optimizer": "Adam",
                "n_trials_run": len(study.trials),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()