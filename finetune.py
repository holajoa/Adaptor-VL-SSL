import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import pytorch_lightning.callbacks as cb

from mgca.datasets.classification_dataset import RSNAImageDataset, COVIDXImageDataset

from models.pipeline import AdaptorPipelineWithClassificationHead
from models.configurations import TEXT_PRETRAINED, VISION_PRETRAINED
from utils.model_utils import get_newest_ckpt, StreamingProgressBar
from utils.dataset_utils import torch2huggingface_dataset, get_dataloader
from dataset.dataset import clf_collator
from utils.args import get_train_parser
from utils.utils import set_environment_for_aml

from math import ceil
import os
import wandb


def main(args):
    if args.vision_model not in VISION_PRETRAINED.keys():
        raise ValueError(f'Vision model {args.vision_model} not available.'
                         f'Choose from {list(VISION_PRETRAINED.keys())}')
        
    if args.text_model not in TEXT_PRETRAINED.keys():
        raise ValueError(f'Text model {args.text_model} not available.'
                         f'Choose from {list(TEXT_PRETRAINED.keys())}')
    
    vision_model_config = VISION_PRETRAINED[args.vision_model]
    args.vision_pretrained = vision_model_config['pretrained_weight']
    args.vision_model_type = vision_model_config['vision_model_type']
    args.vision_output_dim = vision_model_config['vision_output_dim']
    data_transform = vision_model_config['data_transform']
    args.text_pretrained = TEXT_PRETRAINED[args.text_model]
    
    if args.dataset == 'rsna':
        train_dataset = RSNAImageDataset(
            split='train', 
            transform=data_transform(True, args.crop_size), 
            phase='classification', 
            data_pct=args.data_pct, 
            imsize=args.crop_size, 
        )
        val_dataset = RSNAImageDataset(
            split='valid', 
            transform=data_transform(False, args.crop_size), 
            phase='classification', 
            data_pct=args.data_pct, 
            imsize=args.crop_size, 
        )
    elif args.dataset == 'covidx':
        train_dataset = COVIDXImageDataset(
            split='train', 
            transform=data_transform(True, args.crop_size), 
            data_pct=1., 
            imsize=args.crop_size, 
        )
        val_dataset = COVIDXImageDataset(
            split='valid', 
            transform=data_transform(False, args.crop_size), 
            data_pct=args.data_pct, 
            imsize=args.crop_size, 
        )
    else:
        raise ValueError("Must specify dataset - choose between 'covidx' and 'rsna'.")
    
    args.max_steps = ceil(len(train_dataset) / args.batch_size) * args.num_train_epochs
    args.val_steps = ceil(len(val_dataset) / args.batch_size)
    print(f'Number of training samples used: {len(train_dataset)}')
    print(f'Total number of training steps: {args.max_steps}')
    
    train_dataset = torch2huggingface_dataset(train_dataset, streaming=False)
    train_dataset.with_format('torch')
    
    val_dataset = torch2huggingface_dataset(val_dataset, streaming=False)
    val_dataset.with_format('torch')
    
    train_dataloader = get_dataloader(
        train_dataset, 
        batch_size=args.batch_size,
        num_workers=1,
        collate_fn=clf_collator,
    )
    val_dataloader = get_dataloader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=1,
        collate_fn=clf_collator,
    )
    
    model = AdaptorPipelineWithClassificationHead(
        text_model=args.text_model, 
        vision_model=args.vision_model, 
        adaptor_ckpt=get_newest_ckpt(args.vision_model, args.text_model), 
        num_classes=1, 
    )
    
    seed_everything(args.seed)
    
    callbacks = [StreamingProgressBar(total=args.max_steps//args.num_train_epochs, val_total=args.val_steps)]
    
    if args.wandb:
        wandb.login(key='b0236e7bef7b6a3789ca4f305406ab358812da3d')
        logger = WandbLogger(log_model="all", save_dir=args.output_dir, job_type="train")
        logger.watch(model, log_freq=max(100, args.log_every_n_steps))
        logger.log_hyperparams(vars(args))
        experiment_dir = logger.experiment.dir
        callbacks += [
            cb.LearningRateMonitor(), 
            cb.ModelCheckpoint(monitor="train_auroc_step", mode="max"), 
            cb.ModelCheckpoint(monitor="val_auroc_epoch", mode="max"), 
        ]
    else:
        logger = CSVLogger(args.output_dir)
    
    trainer = Trainer(
        accelerator="gpu", 
        devices=args.n_gpus, 
        num_nodes=1, 
        strategy="ddp", 
        # accelerator="cpu",
        # limit_train_batches=1, 
        # limit_val_batches=1, 
        max_epochs=args.num_train_epochs,
        log_every_n_steps=args.log_every_n_steps, 
        check_val_every_n_epoch=1, 
        default_root_dir=args.output_dir,
        callbacks=callbacks,
        enable_progress_bar=False, 
        logger=logger, 
    )
    model.training_steps = args.max_steps
    model.validation_steps = args.val_steps
    
    trainer.fit(model, train_dataloader, val_dataloader)
    
  
if __name__ == '__main__':
    parser = get_train_parser()
    parser.add_argument('--dataset', type=str, required=True, help="Choose between 'covidx' and 'rsna'")
    args = parser.parse_args()
    main(args)
    