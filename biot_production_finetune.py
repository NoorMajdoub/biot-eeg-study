"""Production BIOT fine-tuning run using the winning hyperparameters found by

biot_optuna_search.py -- unlike the search, this DOES report real

test_acc/test_bacc/test_pr_auc/test_auroc (trainer.test() is called), and

runs across the full scaling grid: all 3 data fractions (100/50/25%) x all

3 seeds (1, 2, 42) x both arms, for each of the 6 binary BIOT datasets.



Same non-invasive approach as the search script: imports LitModel_finetune

from run_binary_supervised.py unmodified, subclasses it only to add the

winning warmup+cosine scheduler and (optional) gradient clipping.



HPARAMS below are the winning values from all 36 completed Optuna studies

(chbmit, clinical, dream, karaone, yang2025, japaneeg x pretrained/random

x 100%/50%/25% fraction). batch_size/weight_decay are fixed constants

shared with the search phase (see DATASET_CONFIG for why chbmit=512 while

the rest are smaller).



Usage (one run per invocation -- see submit_biot_production_job.sh):

  python biot_production_finetune.py --dataset chbmit --arm pretrained --fraction 50 --seed 1

"""

import argparse

import json

import math

import os

import sys



import numpy as np

import pytorch_lightning as pl

import torch

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



# Same dataset config as biot_optuna_search.py -- batch_size fixed, matches

# biot_scaling_job.sbatch exactly.

# NOTE: yang2025's root was fixed here (was pointing at a Trillium-only

# nested path, "/scratch/nourmaj/scratch_iteration2/yang2025_processed",

# that never existed on Nibi -- corrected to match the flat convention

# used by every other dataset, matching biot_optuna_search.py).

DATASET_CONFIG = {

    "chbmit": dict(

        BIOT_NAME="CHB-MIT", loader=prepare_CHB_MIT_dataloader,

        in_channels=16, n_classes=1, sample_length=10,

        root="/scratch/nourmaj/chb-mit-processed-80h", batch_size=512, token_size=200, hop_length=100,

    ),

    "clinical": dict(

        BIOT_NAME="Clinical", loader=prepare_Clinical_dataloader,

        in_channels=19, n_classes=1, sample_length=10,

        root="/scratch/nourmaj/ds004504_processed", batch_size=32, token_size=200, hop_length=100,

    ),

    "dream": dict(

        BIOT_NAME="Dream", loader=prepare_Dream_dataloader,

        in_channels=257, n_classes=1, sample_length=20,

        root="/scratch/nourmaj/dream_processed", batch_size=16, token_size=200, hop_length=100,

    ),

    "karaone": dict(

        BIOT_NAME="KaraOne", loader=prepare_KaraOne_dataloader,

        in_channels=60, n_classes=1, sample_length=5,

        root="/scratch/nourmaj/karaone_processed", batch_size=32, token_size=200, hop_length=100,

    ),

    "yang2025": dict(

        BIOT_NAME="Yang2025", loader=prepare_Yang2025_dataloader,

        in_channels=59, n_classes=1, sample_length=6,

        root="/scratch/nourmaj/yang2025_processed", batch_size=32, token_size=200, hop_length=100,

    ),

    "japaneeg": dict(

        BIOT_NAME="JapanEEG", loader=prepare_JapanEEG_dataloader,

        in_channels=128, n_classes=1, sample_length=10,

        root="/scratch/nourmaj/japaneeg_processed_80h", batch_size=32, token_size=200, hop_length=100,

    ),

    "thingseeg1": dict(
        BIOT_NAME="ThingsEEG1", loader=prepare_ThingsEEG1_dataloader,
        in_channels=63, n_classes=1, sample_length=0.5,
        root="/scratch/nourmaj/things_eeg1/splits",
        batch_size=32, token_size=20, hop_length=10,
    ),
}



# Winning hyperparameters from all 36 completed Optuna studies (12 per

# fraction x 3 fractions). Keyed by (dataset, arm, fraction). grad_clip:

# None means "no clipping" (that's what use_grad_clip=False resolved to

# during the search). weight_decay is fixed 1e-5 for everyone -- not part

# of this table since it never varies.

#

# The 50pct karaone/pretrained and 25pct karaone/pretrained entries, and

# the 50pct/25pct yang2025/random entries, were re-run after fixing a bug

# in biot_optuna_search.py's OptunaTrackingCallback where the pre-training

# Lightning "sanity check" validation pass was being counted as if it were

# a real epoch, contaminating the trial's best score for datasets whose

# real training never clearly beat that untrained baseline. All 4 reruns

# were verified against their own per-epoch history (reported == real

# epoch max) before being recorded here. Every other entry was audited the

# same way and confirmed unaffected by that bug.

HPARAMS = {

    # ---- 100% (original round) ----

    ("chbmit", "random", "100"):     dict(lr=0.003409949523456896,  warmup_ratio=0.07256322737389395,  grad_clip=4.272156703326215),

    ("chbmit", "pretrained", "100"): dict(lr=0.00019744099634754297, warmup_ratio=0.018175455832783814, grad_clip=0.43760160075838056),

    ("clinical", "random", "100"):     dict(lr=0.0035970899879300845, warmup_ratio=0.049508354609026625, grad_clip=1.056361732095292),

    ("clinical", "pretrained", "100"): dict(lr=0.00011618599086196344, warmup_ratio=0.08442724786157611, grad_clip=4.420329585538908),

    ("yang2025", "random", "100"):     dict(lr=0.00081233377597827,  warmup_ratio=0.14523696994,          grad_clip=0.9842926085066147),

    ("yang2025", "pretrained", "100"): dict(lr=0.0002815972079427622, warmup_ratio=0.0755210886058506,    grad_clip=2.6114148860121214),

    ("karaone", "random", "100"):     dict(lr=0.0007436696172809815, warmup_ratio=0.061914477742692736, grad_clip=1.7724670369835565),

    ("karaone", "pretrained", "100"): dict(lr=0.00020503819111914143, warmup_ratio=0.07613638480088317, grad_clip=1.123869718967685),

    ("japaneeg", "random", "100"):     dict(lr=0.0012692193293839717, warmup_ratio=0.1196514034529834,  grad_clip=0.96473500186711),

    ("japaneeg", "pretrained", "100"): dict(lr=0.00017077900206027475, warmup_ratio=0.0296200232194812, grad_clip=2.67371615517719),

    ("dream", "random", "100"):     dict(lr=0.002215027827027308,  warmup_ratio=0.0005839084321372682, grad_clip=2.8428182183644535),

    ("dream", "pretrained", "100"): dict(lr=0.0004779801214396841, warmup_ratio=0.029374562095057052,  grad_clip=2.306836086505519),



    # ---- 50% ----

    ("chbmit", "random", "50"):     dict(lr=0.0014631071230124567, warmup_ratio=0.027127396397657457, grad_clip=1.571575520979762),

    ("chbmit", "pretrained", "50"): dict(lr=0.0001254572807294892, warmup_ratio=0.0217622229122579,   grad_clip=0.4965262907),

    ("clinical", "random", "50"):     dict(lr=0.004081996370557849,  warmup_ratio=0.0245010416562567,   grad_clip=None),

    ("clinical", "pretrained", "50"): dict(lr=0.0004962694964841057, warmup_ratio=0.1067476318477309,   grad_clip=0.1069452553761746),

    ("dream", "random", "50"):     dict(lr=0.0035355090715153666, warmup_ratio=0.08743512676667829, grad_clip=0.106363785538376),

    ("dream", "pretrained", "50"): dict(lr=0.000484106704715732,  warmup_ratio=0.0362373563516786,  grad_clip=None),

    ("japaneeg", "random", "50"):     dict(lr=0.0014309402020028959, warmup_ratio=0.070267345220789,    grad_clip=None),

    ("japaneeg", "pretrained", "50"): dict(lr=0.00023397410599353618, warmup_ratio=0.04090590500443539, grad_clip=0.5095780339576),

    ("yang2025", "random", "50"):     dict(lr=0.0014614777381070,    warmup_ratio=0.027765358138,       grad_clip=0.18746313898608),

    ("yang2025", "pretrained", "50"): dict(lr=0.00011553823033741939, warmup_ratio=0.1451128480024598,  grad_clip=None),

    ("karaone", "random", "50"):     dict(lr=0.0007178682828383515, warmup_ratio=0.114484407655711,    grad_clip=None),

    ("karaone", "pretrained", "50"): dict(lr=0.0001257197524201854, warmup_ratio=0.00954412378388553,  grad_clip=None),



    # ---- 25% ----

    ("chbmit", "random", "25"):     dict(lr=0.00236687069558311,  warmup_ratio=0.10570727955717807, grad_clip=0.141396502361628),

    ("chbmit", "pretrained", "25"): dict(lr=0.000252391289373653, warmup_ratio=0.12140511029091859, grad_clip=0.1619872730075643),

    ("clinical", "random", "25"):     dict(lr=0.00184665572481664,  warmup_ratio=0.044631200147546,   grad_clip=0.204715923480415),

    ("clinical", "pretrained", "25"): dict(lr=0.00027596553388304,  warmup_ratio=0.063029608118472,   grad_clip=None),

    ("dream", "random", "25"):     dict(lr=0.0048639900726767,   warmup_ratio=0.062160404479316,   grad_clip=0.1434861170253),

    ("dream", "pretrained", "25"): dict(lr=0.00028574848617153,  warmup_ratio=0.04991541683806513, grad_clip=0.256992137693411),

    ("karaone", "random", "25"):     dict(lr=0.00085275406408978, warmup_ratio=0.108216570395352,   grad_clip=None),

    ("karaone", "pretrained", "25"): dict(lr=0.000365991731455917, warmup_ratio=0.0317686073249901, grad_clip=None),

    ("japaneeg", "random", "25"):     dict(lr=0.001620178781341095, warmup_ratio=0.1152782635397892, grad_clip=1.116730873256953),

    ("japaneeg", "pretrained", "25"): dict(lr=0.00037317092639920,  warmup_ratio=0.123834015287055,  grad_clip=None),

    ("yang2025", "random", "25"):     dict(lr=0.00099764933232749,  warmup_ratio=0.02656732006842,   grad_clip=0.59332182270769),

    ("yang2025", "pretrained", "25"): dict(lr=0.0001364581503421154, warmup_ratio=0.1037901928681937, grad_clip=None),
    ("thingseeg1", "pretrained", "100"): dict(lr=3e-4, warmup_ratio=0.1, grad_clip=None, pos_weight=10.0),
    ("thingseeg1", "random", "100"):     dict(lr=3e-4, warmup_ratio=0.1, grad_clip=None, pos_weight=10.0),
    ("thingseeg1", "pretrained", "50"):  dict(lr=3e-4, warmup_ratio=0.1, grad_clip=None, pos_weight=10.0),
    ("thingseeg1", "random", "50"):      dict(lr=3e-4, warmup_ratio=0.1, grad_clip=None, pos_weight=10.0),
    ("thingseeg1", "pretrained", "25"):  dict(lr=3e-4, warmup_ratio=0.1, grad_clip=None, pos_weight=10.0),
    ("thingseeg1", "random", "25"):      dict(lr=3e-4, warmup_ratio=0.1, grad_clip=None, pos_weight=10.0),
}





class LitModel_finetune_tuned(LitModel_finetune):

    """Same reasoning as biot_optuna_search.py's LitModel_finetune_optuna:

    configure_optimizers adds the winning warmup+cosine scheduler;

    validation_step/test_step get a .float() cast before .cpu().numpy()

    since real bf16 precision (this Lightning version) breaks numpy

    conversion otherwise. Everything else inherited unchanged."""



    def __init__(self, args, model, warmup_ratio, pos_weight=None):
        super().__init__(args, model, pos_weight=pos_weight)
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





def build_args(dataset_key, arm, lr, seed):

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

    args.epochs = 15 if dataset_key == "thingseeg1" else 100

    args.lr = lr

    args.weight_decay = 1e-5

    args.batch_size = cfg["batch_size"]

    args.num_workers = 8

    args.seed = seed

    args.pretrain_model_path = PRETRAIN_CKPT if arm == "pretrained" else ""

    return args





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG))

    parser.add_argument("--arm", required=True, choices=["pretrained", "random"])

    parser.add_argument("--fraction", required=True, choices=["100", "50", "25"])

    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])

    cli_args = parser.parse_args()



    cfg = DATASET_CONFIG[cli_args.dataset]

    hp = HPARAMS[(cli_args.dataset, cli_args.arm, cli_args.fraction)]



    print(f"=== BIOT production run: dataset={cli_args.dataset} arm={cli_args.arm} "

          f"fraction={cli_args.fraction} seed={cli_args.seed} ===")

    print(f"tuned hyperparameters: {hp}")



    args = build_args(cli_args.dataset, cli_args.arm, hp["lr"], cli_args.seed)



    torch.manual_seed(args.seed)

    np.random.seed(args.seed)



    if cli_args.fraction == "100":

        data_root = cfg["root"]

    else:

        data_root = f"{cfg['root']}_{cli_args.fraction}pct"

    if not os.path.isdir(data_root):

        print(f"ERROR: expected data folder not found: {data_root}")

        sys.exit(1)

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



        lightning_model = LitModel_finetune_tuned(args, model, hp["warmup_ratio"], pos_weight=hp.get("pos_weight"))



        patience = 3 if cli_args.dataset == "thingseeg1" else 5
        early_stop_callback = EarlyStopping(
        monitor="val_auroc", patience=patience, verbose=False, mode="max"
         )



    grad_clip = hp["grad_clip"]  # None means no clipping



    # Lightning's default checkpoint dir is based on the CWD (which

    # biot_production_job.sbatch sets to /home/nourmaj/BIOT), not $HOME --

    # so it isn't covered by this project's usual $HOME->scratch redirect.

    # Point it at scratch explicitly to avoid writing under /home.

    checkpoint_dir = (

        f"/scratch/nourmaj/biot_checkpoints/"

        f"{cli_args.dataset}_{cli_args.arm}_{cli_args.fraction}pct_s{cli_args.seed}")

    os.makedirs(checkpoint_dir, exist_ok=True)



    trainer = pl.Trainer(

        devices=[0],

        accelerator="gpu",

        benchmark=True,

        enable_checkpointing=True,

        default_root_dir=checkpoint_dir,

        logger=False,

        max_epochs=args.epochs,

        gradient_clip_val=grad_clip,

        precision="bf16",

        callbacks=[early_stop_callback],

    )



    trainer.fit(

        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader

    )



    test_result = trainer.test(

        model=lightning_model, ckpt_path="best", dataloaders=test_loader

    )[0]

    print("=== TEST RESULT ===")

    print(test_result)



    out_dir = "/scratch/nourmaj/biot_production_results"

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

                "batch_size": cfg["batch_size"],

                "weight_decay": 1e-5,

                "test_result": test_result,

            },

            f,

            indent=2,

        )

    print(f"Saved: {out_path}")





if __name__ == "__main__":

    main()

