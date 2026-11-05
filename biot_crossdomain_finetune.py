"""
Cross-domain transfer test: load a BIOT ENCODER that was fine-tuned on one
dataset (the "source", e.g. Clinical), discard its classification head,
attach a FRESH head sized for a different "target" dataset from a different
domain, and fine-tune end-to-end on the target. Goal: see whether knowledge
learned on the source domain transfers to the target domain.

This is exploratory ("testing the water") -- NOT Optuna-tuned. Uses a single
reasonable set of hyperparameters per run rather than a search.

Usage:
    python biot_crossdomain_finetune.py \
        --source_ckpt /scratch/nourmaj/biot_checkpoints/clinical_pretrained_100pct_s1/checkpoints/epoch=26-step=459.ckpt \
        --target_dataset isruc \
        --seed 0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

sys.path.insert(0, "/home/nourmaj/BIOT")

from model import BIOTClassifier  # noqa: E402

# Both run_*_supervised.py files define a class literally named
# LitModel_finetune -- alias on import so there's no collision.
from run_multiclass_supervised import (  # noqa: E402
    LitModel_finetune as LitModel_finetune_mc,
    prepare_ISRUC_dataloader,
)
from run_binary_supervised import (  # noqa: E402
    LitModel_finetune as LitModel_finetune_bin,
    prepare_Yang2025_dataloader,
    prepare_KaraOne_dataloader,
)

TARGET_CONFIG = {
    "isruc": dict(
        pipeline="multiclass", BIOT_NAME="ISRUC", loader=prepare_ISRUC_dataloader,
        in_channels=6, n_classes=5, sample_length=30, sampling_rate=200,
        token_size=200, hop_length=100, batch_size=32, monitor="val_cohen",
    ),
    "yang2025": dict(
        pipeline="binary", BIOT_NAME="Yang2025", loader=prepare_Yang2025_dataloader,
        in_channels=59, n_classes=1, sample_length=6, sampling_rate=200,
        token_size=200, hop_length=100, batch_size=32, monitor="val_auroc",
    ),
    "karaone": dict(
        pipeline="binary", BIOT_NAME="KaraOne", loader=prepare_KaraOne_dataloader,
        in_channels=60, n_classes=1, sample_length=5, sampling_rate=200,
        token_size=200, hop_length=100, batch_size=32, monitor="val_auroc",
    ),
}


def load_source_encoder(model, source_ckpt_path):
    """Load only the BIOT encoder weights from a source checkpoint, which
    may be either:
      (a) a full PyTorch Lightning checkpoint from LitModel_finetune, where
          encoder weights live under state_dict["model.biot.*"], or
      (b) a raw encoder-only checkpoint (BIOT's official release format),
          where the top-level keys already match model.biot's own keys.
    Shape-mismatched keys (different channel count between source and
    target dataset) are skipped and left randomly initialized, same as the
    existing pretrained-checkpoint loading logic in run_*_supervised.py.
    """
    ckpt = torch.load(source_ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw_sd = ckpt["state_dict"]
        prefix = "model.biot."
        source_sd = {k[len(prefix):]: v for k, v in raw_sd.items() if k.startswith(prefix)}
        if not source_sd:
            sample_keys = list(raw_sd.keys())[:8]
            raise RuntimeError(
                f"No '{prefix}' keys found in checkpoint's state_dict. "
                f"Sample keys actually present: {sample_keys}"
            )
        print(f"Detected PyTorch Lightning checkpoint; extracted {len(source_sd)} encoder keys under '{prefix}'")
    else:
        source_sd = ckpt
        print(f"Detected raw encoder-only checkpoint with {len(source_sd)} keys")

    model_state = model.biot.state_dict()
    skip_keys = [k for k in source_sd if k in model_state and source_sd[k].shape != model_state[k].shape]
    if skip_keys:
        print(f"Skipping {len(skip_keys)} mismatched source-encoder keys (channel-count dependent, reinitialized randomly):")
        for k in skip_keys:
            print(f"    {k}: source={source_sd[k].shape} target={model_state[k].shape}")

    filtered = {k: v for k, v in source_sd.items() if k not in skip_keys}
    missing, unexpected = model.biot.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)} matched keys from source encoder. Missing in target: {len(missing)}. Unexpected: {len(unexpected)}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_ckpt", required=True, help="path to the source-domain fine-tuned .ckpt")
    parser.add_argument("--target_dataset", required=True, choices=list(TARGET_CONFIG))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--freeze_encoder", action="store_true",
                         help="linear-probe mode: freeze the transferred encoder, only train the new head")
    cli_args = parser.parse_args()

    cfg = TARGET_CONFIG[cli_args.target_dataset]

    args = argparse.Namespace()
    args.dataset = cfg["BIOT_NAME"]
    args.in_channels = cfg["in_channels"]
    args.n_classes = cfg["n_classes"]
    args.sample_length = cfg["sample_length"]
    args.sampling_rate = cfg["sampling_rate"]
    args.token_size = cfg["token_size"]
    args.hop_length = cfg["hop_length"]
    args.epochs = cli_args.epochs
    args.lr = cli_args.lr
    args.weight_decay = cli_args.weight_decay
    args.batch_size = cfg["batch_size"]
    args.num_workers = cli_args.num_workers
    args.seed = cli_args.seed

    print(f"=== BIOT cross-domain transfer: source={cli_args.source_ckpt} -> target={cli_args.target_dataset} seed={cli_args.seed} ===")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, test_loader, val_loader = cfg["loader"](args)

    model = BIOTClassifier(
        n_classes=args.n_classes,
        n_channels=args.in_channels,
        n_fft=args.token_size,
        hop_length=args.hop_length,
    )
    model = load_source_encoder(model, cli_args.source_ckpt)

    if cli_args.freeze_encoder:
        for p in model.biot.parameters():
            p.requires_grad = False
        print("Encoder frozen (linear-probe mode) -- only the new head will train")

    LitClass = LitModel_finetune_mc if cfg["pipeline"] == "multiclass" else LitModel_finetune_bin
    lightning_model = LitClass(args, model)

    early_stop_callback = EarlyStopping(
        monitor=cfg["monitor"], patience=5, verbose=False, mode="max"
    )

    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        benchmark=True,
        enable_checkpointing=False,
        logger=False,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback],
    )

    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_result = trainer.test(model=lightning_model, dataloaders=test_loader)[0]
    print(test_result)

    out_dir = "/scratch/nourmaj/biot_crossdomain_results"
    os.makedirs(out_dir, exist_ok=True)
    source_tag = os.path.basename(os.path.dirname(os.path.dirname(cli_args.source_ckpt)))  # e.g. clinical_pretrained_100pct_s1
    out_path = f"{out_dir}/{source_tag}_TO_{cli_args.target_dataset}_s{cli_args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "source_ckpt": cli_args.source_ckpt,
                "target_dataset": cli_args.target_dataset,
                "freeze_encoder": cli_args.freeze_encoder,
                "seed": cli_args.seed,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs_requested": args.epochs,
                "test_result": test_result,
            },
            f,
            indent=2,
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
