"""
Production BIOT fine-tuning for the MULTICLASS datasets (isruc, thoughtviz).
Loads winning hyperparameters from the completed Optuna search into
HPARAMS below (fill in after biot_optuna_search_multiclass.py finishes
each combo), fine-tunes with the warmup+cosine schedule, and reports real
test metrics (test_acc, test_cohen, test_f1 -- multiclass's equivalent of
binary's test_acc/test_bacc/test_pr_auc/test_auroc).
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

sys.path.insert(0, "/home/nourmaj/BIOT")
from run_multiclass_supervised import (  # noqa: E402
    LitModel_finetune,
    prepare_ISRUC_dataloader,
    prepare_ThoughtViz_dataloader,
    prepare_CAUEEG_dataloader,
    prepare_SleepEDF_dataloader,
)
from model import BIOTClassifier  # noqa: E402

PRETRAIN_CKPT = "pretrained-models/EEG-SHHS+PREST-18-channels.ckpt"

DATASET_CONFIG = {

    "sleepedf": dict(
        BIOT_NAME="SleepEDF", loader=prepare_SleepEDF_dataloader,
        in_channels=2, n_classes=5, sample_length=30, sampling_rate=200,
        token_size=200, hop_length=100,
        root="/scratch/nourmaj/sleepedf_reve_processed",
        batch_size=32,
    ),
    "isruc": dict(
        BIOT_NAME="ISRUC", loader=prepare_ISRUC_dataloader,
        in_channels=6, n_classes=5, sample_length=30, sampling_rate=200,
        token_size=200, hop_length=100,
        root="/scratch/nourmaj/isruc_biot_processed",
        batch_size=32,
    ),
    "thoughtviz": dict(
        BIOT_NAME="ThoughtViz", loader=prepare_ThoughtViz_dataloader,
        in_channels=14, n_classes=10, sample_length=0.25, sampling_rate=200,
        token_size=20, hop_length=10,
        root="/scratch/nourmaj/thoughtviz_processed",
        batch_size=32,
    ),
    "caueeg": dict(
        BIOT_NAME="CAUEEG", loader=prepare_CAUEEG_dataloader,
        in_channels=19, n_classes=3, sample_length=10, sampling_rate=200,
        token_size=200, hop_length=100,   # standard settings, same as isruc -- 2000-sample windows are plenty long
        root="/scratch/nourmaj/caueeg/caueeg-dataset/splits",
        batch_size=32,
    ),
}

# Done: we filled this manually from the json resutls
# biot_optuna_search_multiclass.py finishes each combo -- 12 entries total
# (2 datasets x 2 arms x 3 fractions).
HPARAMS = {

    ("sleepedf", "pretrained", "100"): {"lr": 0.000491821207325646, "warmup_ratio": 0.11051684770832855},
    ("sleepedf", "random", "100"):     {"lr": 0.0005564802589423432, "grad_clip": 4.922386913304661, "warmup_ratio": 0.03200848019503561},
    ("isruc", "pretrained", "100"): {"lr": 0.000491821207325646, "warmup_ratio": 0.11051684770832855},
    ("isruc", "pretrained", "50"):  {"lr": 0.00021654810401804307, "warmup_ratio": 0.04017123078926643},
    ("isruc", "pretrained", "25"):  {"lr": 0.00025277095974757827, "grad_clip": 2.848706599292819, "warmup_ratio": 0.07709085857701374},
    ("isruc", "random", "100"):     {"lr": 0.0005564802589423432, "grad_clip": 4.922386913304661, "warmup_ratio": 0.03200848019503561},
    ("isruc", "random", "50"):      {"lr": 0.0005248461435798516, "warmup_ratio": 0.03473137010820143},
    ("isruc", "random", "25"):      {"lr": 0.0005267762996120661, "grad_clip": 1.5019676009721594, "warmup_ratio": 0.00033259253499405214},

    ("thoughtviz", "pretrained", "100"): {"lr": 0.00012565645484653558, "warmup_ratio": 0.06290367050631114},
    ("thoughtviz", "pretrained", "50"):  {"lr": 0.0002663272505493497, "warmup_ratio": 0.09928258869927903},
    ("thoughtviz", "pretrained", "25"):  {"lr": 0.00024036695246008608, "grad_clip": 2.6404220133402365, "warmup_ratio": 0.05896079946028201},
    ("thoughtviz", "random", "100"):     {"lr": 0.004758826682323033, "warmup_ratio": 0.09800961684425999},
    ("thoughtviz", "random", "50"):      {"lr": 0.0007368011352376395, "grad_clip": 1.3722109062678043, "warmup_ratio": 0.08466865994462043},
    ("thoughtviz", "random", "25"):      {"lr": 0.004841786347194247, "grad_clip": 3.3196568244580047, "warmup_ratio": 0.1297853159056631},
    
       ("caueeg", "pretrained", "100"): {"lr": 0.0003238889796989414, "grad_clip": 0.47753483462792595, "warmup_ratio": 0.05040258507233329},
    ("caueeg", "random", "100"):     {"lr": 0.003358889806368957, "grad_clip": 3.0142168037007036, "warmup_ratio": 0.03406373986861964},
    ("caueeg", "pretrained", "50"):  {"lr": 0.0002484345481569901, "grad_clip": 0.10189081466542228, "warmup_ratio": 0.13876085880193342},
    ("caueeg", "random", "50"):      {"lr": 0.004131166422130799, "warmup_ratio": 0.04104379634782999},
    ("caueeg", "pretrained", "25"):  {"lr": 0.000469881266328136, "warmup_ratio": 0.06606622480910623},
    ("caueeg", "random", "25"):      {"lr": 0.002957787220175301, "warmup_ratio": 0.1052371197034676},
    
}

class LitModel_finetune_tuned(LitModel_finetune):
    """Adds the warmup+cosine scheduler. No bf16 cast override needed --
    this trainer doesn't request bf16 precision."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG))
    parser.add_argument("--arm", required=True, choices=["pretrained", "random"])
    parser.add_argument("--fraction", required=True, choices=["100", "50", "25"])
    parser.add_argument("--seed", type=int, required=True)
    cli_args = parser.parse_args()

    cfg = DATASET_CONFIG[cli_args.dataset]
    hp = HPARAMS[(cli_args.dataset, cli_args.arm, cli_args.fraction)]

    args = argparse.Namespace()
    args.dataset = cfg["BIOT_NAME"]
    args.model = "BIOT"
    args.in_channels = cfg["in_channels"]
    args.n_classes = cfg["n_classes"]
    args.sample_length = cfg["sample_length"]
    args.sampling_rate = cfg["sampling_rate"]
    args.token_size = cfg["token_size"]
    args.hop_length = cfg["hop_length"]
    args.epochs = 100
    args.lr = hp["lr"]
    args.weight_decay = 1e-5
    args.batch_size = cfg["batch_size"]
    args.num_workers = 8
    args.seed = cli_args.seed
    args.pretrain_model_path = PRETRAIN_CKPT if cli_args.arm == "pretrained" else ""

    print(f"=== BIOT multiclass production run: dataset={cli_args.dataset} arm={cli_args.arm} fraction={cli_args.fraction} seed={cli_args.seed} ===")
    print(f"tuned hyperparameters: {hp}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if cli_args.fraction == "100":
        data_root = cfg["root"]
    else:
        data_root = f"{cfg['root']}_{cli_args.fraction}pct"
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

    lightning_model = LitModel_finetune_tuned(args, model, hp["warmup_ratio"])

    early_stop_callback = EarlyStopping(
        monitor="val_cohen", patience=5, verbose=False, mode="max"
    )
    grad_clip = hp.get("grad_clip")  # None if use_grad_clip was False during search

    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        benchmark=True,
        enable_checkpointing=False,
        logger=False,
        max_epochs=args.epochs,
        gradient_clip_val=grad_clip,
        callbacks=[early_stop_callback],
    )

    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_result = trainer.test(model=lightning_model, dataloaders=test_loader)[0]
    print(test_result)

    out_dir = "/scratch/nourmaj/biot_production_results_multiclass"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{cli_args.dataset}_{cli_args.arm}_{cli_args.fraction}pct_s{cli_args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": cli_args.dataset,
                "arm": cli_args.arm,
                "fraction": cli_args.fraction,
                "seed": cli_args.seed,
                "hparams": hp,
                "batch_size": args.batch_size,
                "weight_decay": args.weight_decay,
                "test_result": test_result,
            },
            f,
            indent=2,
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()