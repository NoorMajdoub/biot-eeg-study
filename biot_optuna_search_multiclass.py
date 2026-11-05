"""
Per-(dataset, arm) Optuna hyperparameter search for BIOT fine-tuning on the
MULTICLASS datasets (isruc, thoughtviz) -- these use
run_multiclass_supervised.py's LitModel_finetune (CrossEntropyLoss,
val_acc/val_cohen/val_f1), NOT run_binary_supervised.py's (BCE/val_auroc).
Mirrors biot_optuna_search.py's structure and already includes the
sanity-checking guard fix found in that script (see
OptunaTrackingCallback below) -- no bf16 cast override is applied here
since run_multiclass_supervised.py's own supervised() doesn't request
bf16 precision either; add it later if speed becomes an issue.

TODO before running: confirm in_channels (ISRUC=6, confirmed from this
repo's own channel-mismatch comment; ThoughtViz=? not yet confirmed),
n_classes (ISRUC sleep-stage count -- 5 or 6 depending on whether
"Movement" is a separate class; ThoughtViz=?), sample_length, sampling_rate,
and batch_size for both datasets below.
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

from run_multiclass_supervised import (  # noqa: E402
    LitModel_finetune,
    prepare_ISRUC_dataloader,
    prepare_ThoughtViz_dataloader,
    prepare_CAUEEG_dataloader,
)
from model import BIOTClassifier  # noqa: E402

PRETRAIN_CKPT = "pretrained-models/EEG-SHHS+PREST-18-channels.ckpt"

# TODO: confirm every field below against the real dataset before running.
DATASET_CONFIG = {
    "isruc": dict(
        BIOT_NAME="ISRUC", loader=prepare_ISRUC_dataloader,
        in_channels=6, n_classes=5, sample_length=30, sampling_rate=200,
        token_size=200, hop_length=100,   # BIOT's usual defaults -- fine here
        root="/scratch/nourmaj/isruc_biot_processed",
        batch_size=32,
    ),
    "thoughtviz": dict(
        BIOT_NAME="ThoughtViz", loader=prepare_ThoughtViz_dataloader,
        in_channels=14, n_classes=10, sample_length=0.25, sampling_rate=200,
        # 50 samples total after resampling -- token_size must be well under
        # that. token_size=20, hop_length=10 gives floor((50-20)/10)+1 = 4
        # tokens per sample, which is at least something for BIOT's
        # transformer to attend over.
        token_size=20, hop_length=10,
        root="/scratch/nourmaj/thoughtviz_processed",
        batch_size=32,
    ),
    "caueeg": dict(
        BIOT_NAME="CAUEEG", loader=prepare_CAUEEG_dataloader,
        in_channels=19, n_classes=3, sample_length=10, sampling_rate=200,
        token_size=200, hop_length=100,  
        root="/scratch/nourmaj/caueeg/caueeg-dataset/splits",
        batch_size=32,
    ),
    
}

LR_RANGE = {
    "pretrained": (1e-4, 5e-4),
    "random": (5e-4, 5e-3),
}


class LitModel_finetune_optuna(LitModel_finetune):
    """Same as LitModel_finetune, only configure_optimizers is overridden
    to add a linear-warmup-then-cosine-decay scheduler, matching the binary
    search script's approach. validation_epoch_end/test_epoch_end are left
    untouched -- no bf16 cast issue here since precision stays default."""

    def __init__(self, args, model, warmup_ratio):
        super().__init__(args, model)
        self.warmup_ratio = warmup_ratio

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
    """Reports val_cohen to Optuna after every epoch (the multiclass
    equivalent of val_auroc -- matches production's own EarlyStopping
    monitor) and tracks the BEST val_cohen seen across all epochs.
    Includes the sanity-checking guard from day one -- this is the exact
    bug found and fixed in biot_optuna_search.py's binary version; no
    reason to reintroduce it here."""

    def __init__(self, trial, monitor="val_cohen"):
        self.trial = trial
        self.monitor = monitor
        self.best_score = -1.0  # cohen_kappa can be negative, unlike auroc

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
    args.sampling_rate = cfg["sampling_rate"]
    args.token_size = cfg["token_size"]     # was hardcoded 200
    args.hop_length = cfg["hop_length"]     # was hardcoded 100
    args.epochs = 100
    args.lr = lr
    args.weight_decay = 1e-5
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

    if fraction == "100":
        data_root = cfg["root"]
    else:
        data_root = f"{cfg['root']}_{fraction}pct"
    os.environ["BIOT_DATA_ROOT_OVERRIDE"] = data_root
    print(f"Using data root: {data_root}")
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
        monitor="val_cohen", patience=5, verbose=False, mode="max"
    )
    tracking_callback = OptunaTrackingCallback(trial, monitor="val_cohen")

    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        benchmark=True,
        enable_checkpointing=False,
        logger=False,
        max_epochs=args.epochs,
        gradient_clip_val=grad_clip,
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

    out_dir = "/scratch/nourmaj/biot_optuna_results_multiclass"
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{cli_args.dataset}_{cli_args.arm}_{cli_args.fraction}pct"
    db_path = f"{out_dir}/{tag}.db"
    study_name = f"biot_{tag}"

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    sampler = optuna.samplers.TPESampler(seed=None, n_startup_trials=4)

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True,
    )

    n_done = len(
        [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    )
    n_remaining = max(0, cli_args.n_trials - n_done)
    print(f"=== BIOT multiclass Optuna search: dataset={cli_args.dataset} arm={cli_args.arm} fraction={cli_args.fraction} ===")
    print(f"{n_done} trials already recorded in {db_path}, running {n_remaining} more")

    if n_remaining > 0:
        study.optimize(
            lambda trial: objective(trial, cli_args.dataset, cli_args.arm, cli_args.fraction),
            n_trials=n_remaining,
        )

    print("=== BEST TRIAL ===")
    print(f"best val_cohen = {study.best_value}")
    print(f"best params = {study.best_params}")

    with open(f"{out_dir}/{tag}_best.json", "w") as f:
        json.dump(
            {
                "dataset": cli_args.dataset,
                "arm": cli_args.arm,
                "fraction": cli_args.fraction,
                "best_val_cohen": study.best_value,
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