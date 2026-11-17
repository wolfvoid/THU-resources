import argparse
import pathlib
import time

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

from datasets.fusiongallery import FusionGalleryDataset
from uvnet.models import Segmentation

parser = argparse.ArgumentParser("UV-Net solid model face segmentation")
parser.add_argument(
    "traintest", choices=("train", "test"), help="Whether to train or test"
)
parser.add_argument(
    "--dataset", choices=("mfcad", "fusiongallery"), help="Segmentation dataset"
)
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument(
    "--num_workers",
    type=int,
    default=32,
    help="Number of workers for the dataloader. NOTE: set this to 0 on Windows, any other value leads to poor performance",
)
parser.add_argument(
    "--random_rotate",
    action="store_true",
    help="Whether to randomly rotate the solids in 90 degree increments along the canonical axes",
)
parser.add_argument(
    "--crv_in_channels",
    type=int,
    default=6,
    help="Number of channels for curve input",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint file to load weights from for testing",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="segmentation",
    help="Experiment name (used to create folder inside ./results/ to save logs and checkpoints)",
)
parser.add_argument(
    "--graph_encoder_type",
    type=str,
    default="GNN",
    choices=[
        "GNN"
    ],
)
parser.add_argument(
    "--desc",
    type=str,
)
parser.add_argument(
    "--crv_emb_dim",
    type=int,
    default=64,
    help="Dimensionality of the crv embedding",
)
parser.add_argument(
    "--srf_emb_dim",
    type=int,
    default=64,
    help="Dimensionality of the srf embedding",
)
parser.add_argument(
    "--graph_emb_dim",
    type=int,
    default=128,
    help="Dimensionality of the graph embedding",
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.3,
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed to use",
)
parser.add_argument(
    "--accelerator",
    type=str,
    default='gpu',
    help="use gpu or cpu",
)

parser.add_argument(
    "--devices",
    type=int,
    default=1,
)

parser.add_argument(
    "--max_epochs",
    type=int,
    default=50,
)

parser.add_argument(
    "--conv_type",
    type=str,
    default='GCN',
)

args = parser.parse_args()

results_path = (
    pathlib.Path(__file__).parent.joinpath("results").joinpath(args.experiment_name)
)
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

# Define a path to save the results based date and time. E.g.
# results/args.experiment_name/0430/123103
month_day = time.strftime("%m%d")
hour_min_second = time.strftime("%H%M%S") +"_"+ args.desc
if not results_path.joinpath(month_day, hour_min_second).exists():
    results_path.joinpath(month_day, hour_min_second).mkdir(parents=True, exist_ok=True)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=str(results_path.joinpath(month_day, hour_min_second)),
    filename="best",
    save_last=True,
)
trainer = Trainer(
    accelerator=args.accelerator,
    devices=args.devices,
    max_epochs=args.max_epochs,
    callbacks=[checkpoint_callback],
    logger=TensorBoardLogger(
        str(results_path), name=month_day, version=hour_min_second,
    ),
)


Dataset = FusionGalleryDataset

if args.traintest == "train":
    # Train/val
    seed_everything(seed=args.seed, workers=True)
    print(
        f"""
-----------------------------------------------------------------------------------
UV-Net Segmentation
-----------------------------------------------------------------------------------
Logs written to results/{args.experiment_name}/{month_day}/{hour_min_second}

To monitor the logs, run:
tensorboard --logdir results/{args.experiment_name}/{month_day}/{hour_min_second}

The trained model with the best validation loss will be written to:
results/{args.experiment_name}/{month_day}/{hour_min_second}/best.ckpt
-----------------------------------------------------------------------------------
    """
    )
    # write the args to a file(if not exists, create it)
    with open(str(results_path.joinpath(month_day, hour_min_second, "args.json")), "w") as f:
        f.write(str(args))
    model = Segmentation(
        num_classes=Dataset.num_classes(),
        graph_encoder_type=args.graph_encoder_type,
        crv_in_channels=args.crv_in_channels,
        srf_emb_dim=args.srf_emb_dim,
        crv_emb_dim=args.crv_emb_dim,
        graph_emb_dim=args.graph_emb_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        conv_type=args.conv_type,
    )
    # save the model architecture to a file
    with open(str(results_path.joinpath(month_day, hour_min_second, "model_architecture.txt")), "w") as f:
        f.write(str(model))
    train_data = Dataset(
        root_dir=args.dataset_path, split="train", random_rotate=args.random_rotate
    )
    val_data = Dataset(root_dir=args.dataset_path, split="val")
    train_loader = train_data.get_dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = val_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    trainer.fit(model, train_loader, val_loader)
else:
    # Test
    assert (
        args.checkpoint is not None
    ), "Expected the --checkpoint argument to be provided"
    model = Segmentation.load_from_checkpoint(args.checkpoint)
    test_data = Dataset(
        root_dir=args.dataset_path, split="test", random_rotate=args.random_rotate
    )
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    results = trainer.test(model=model, dataloaders=[test_loader], verbose=False)
    print(f"Checkpoint {args.checkpoint}\nSegmentation IoU (%) on test set: {results[0]['test_iou'] * 100.0}")
    print(f"Segmentation Accuracy (%) on test set: {results[0]['test_accuracy'] * 100.0}")
