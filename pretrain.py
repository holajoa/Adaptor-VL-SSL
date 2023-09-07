import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import pytorch_lightning.callbacks as cb

from dataset.dataset import MultimodalPretrainedEmbeddingsDataset
from models.adaptor import Adaptor
from models.configurations import TEXT_PRETRAINED, VISION_PRETRAINED
from utils.dataset_utils import torch2huggingface_dataset, get_dataloader
from utils.model_utils import StreamingProgressBar
from utils.args import get_train_parser

from math import ceil
import wandb


def main(args):
    seed_everything(args.seed, workers=True)
    if args.vision_model not in VISION_PRETRAINED.keys():
        raise ValueError(
            f"Vision model {args.vision_model} not available."
            f"Choose from {list(VISION_PRETRAINED.keys())}"
        )

    if args.text_model not in TEXT_PRETRAINED.keys():
        raise ValueError(
            f"Text model {args.text_model} not available."
            f"Choose from {list(TEXT_PRETRAINED.keys())}"
        )

    vision_model_config = VISION_PRETRAINED[args.vision_model]
    args.vision_pretrained = vision_model_config["pretrained_weight"]
    args.vision_model_type = vision_model_config["vision_model_type"]
    args.vision_output_dim = vision_model_config["vision_output_dim"]
    args.text_pretrained = TEXT_PRETRAINED[args.text_model]

    ### Define model
    model = Adaptor(
        vision_output_dim=args.vision_output_dim,
        projection_dim=args.projection_dim,
        num_layers=args.num_layers,
        lr=args.lr,
    )

    ### Load dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_device = "cpu" if args.num_workers > 0 else device
    train_dataset = MultimodalPretrainedEmbeddingsDataset(
        args.text_model,
        args.vision_model,
        split="train",
        num_of_samples=args.num_of_samples,
        device=dataset_device,
        shuffle=True,
        seed=args.seed,
    )

    val_dataset = MultimodalPretrainedEmbeddingsDataset(
        args.text_model,
        args.vision_model,
        split="valid",
        num_of_samples=args.num_of_samples,
        device=dataset_device,
    )

    args.max_steps = ceil(len(train_dataset) / args.batch_size) * args.num_train_epochs
    args.val_steps = ceil(len(val_dataset) / args.batch_size)
    print(f"Number of training samples used: {len(train_dataset)}")
    print(f"Total number of training steps: {args.max_steps}")

    # Get dataloaders
    train_dataloader = get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=None,
    )
    val_dataloader = get_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=None,
    )

    ### Training
    # seed_everything(args.seed)
    callbacks = [
        StreamingProgressBar(
            total=args.max_steps // args.num_train_epochs, val_total=args.val_steps
        ),
    ]
    if args.wandb:
        wandb.login(key="b0236e7bef7b6a3789ca4f305406ab358812da3d")
        logger = WandbLogger(
            project=f"adaptor_pretrain_{args.num_layers}_layers"
            if not args.project_name
            else args.project_name,
            name=f"{args.vision_model}_{args.text_model}_{args.data_pct}",
            log_model="all",
            save_dir=args.output_dir,
            job_type="train",
        )
        logger.watch(model, log_freq=max(100, args.log_every_n_steps))
        logger.log_hyperparams(vars(args))
        experiment_dir = logger.experiment.dir
        callbacks += [
            cb.LearningRateMonitor(logging_interval="step"),
            cb.ModelCheckpoint(monitor=f"val_loss", mode="min"),
            cb.EarlyStopping(
                monitor="val_loss", patience=10, min_delta=1e-5, mode="min"
            ),
        ]
    else:
        logger = CSVLogger(args.output_dir)

    if args.cpu:
        device_kwargs = {"accelerator": "cpu"}
    else:
        device_kwargs = {
            "accelerator": "gpu",
            "devices": args.n_gpus,
            "num_nodes": 1,
            "strategy": "ddp_find_unused_parameters_false",
        }

    trainer = Trainer(
        max_epochs=args.num_train_epochs,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=1,
        default_root_dir=args.output_dir,
        callbacks=callbacks,
        enable_progress_bar=False,
        logger=logger,
        **device_kwargs,
    )

    model.training_steps = args.max_steps
    model.validation_steps = args.val_steps

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    parser = get_train_parser()
    args = parser.parse_args()
    main(args)
