import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import pytorch_lightning.callbacks as cb

from models.pipeline import AdaptorPipelineWithClassificationHead
from models.configurations import TEXT_PRETRAINED, VISION_PRETRAINED
from utils.model_utils import freeze_adaptor, get_newest_ckpt, StreamingProgressBar, TestEveryEpochCallback
from dataset.dataset import clf_collator
from dataset.configurations import DATASET_CFG  
from dataset.data_module import AdaptorDataModule
from utils.args import get_train_parser

from math import ceil
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
    
    dataset_cfg = DATASET_CFG[args.dataset]
    dataset_class = dataset_cfg['class']
    dataset_kwargs = dataset_cfg['kwargs']

    data_module = AdaptorDataModule(
        dataset=dataset_class, 
        collate_fn=clf_collator, 
        transforms=data_transform,
        data_pct=args.data_pct, 
        batch_size=args.batch_size, 
        num_workers=1,
        crop_size=args.crop_size, 
        seed=args.seed,
        **dataset_kwargs, 
    )
    data_module.setup(stage='fit')
    
    args.max_steps = data_module.train_steps * args.num_train_epochs
    args.val_steps = data_module.val_steps
    # print(f"Number of training samples used: {len(data_module.datasets['train'])}")
    print(f'Total number of training steps: {args.max_steps}')

    model = AdaptorPipelineWithClassificationHead(
        text_model=args.text_model, 
        vision_model=args.vision_model, 
        adaptor_ckpt=get_newest_ckpt(args.vision_model, args.text_model, wandb=args.wandb), 
        num_classes=dataset_cfg['num_classes'], 
        lr=args.lr, 
    )
    if not args.unfreeze_adaptor:
        freeze_adaptor(model)
    
    seed_everything(args.seed, workers=True)
    
    callbacks = [
        StreamingProgressBar(total=data_module.train_steps, val_total=data_module.val_steps), 
    ]
    
    if args.wandb:
        wandb.login(key='b0236e7bef7b6a3789ca4f305406ab358812da3d')
        logger = WandbLogger(log_model="all", save_dir=args.output_dir, job_type="train")
        logger.watch(model, log_freq=max(100, args.log_every_n_steps))
        logger.log_hyperparams(vars(args))
        experiment_dir = logger.experiment.dir
        callbacks += [
            cb.LearningRateMonitor(), 
            cb.ModelCheckpoint(monitor=f"train_{model.metric_name}_step", mode="max"), 
            cb.ModelCheckpoint(monitor=f"val_{model.metric_name}_epoch", mode="max"), 
            cb.EarlyStopping(monitor="val_loss", min_delta=0., patience=args.num_train_epochs//3, verbose=False, mode="min")
        ]
    else:
        logger = CSVLogger(args.output_dir)
    
    if args.cpu:
        device_kwargs = {'accelerator':'cpu'}
    else:
        device_kwargs = {'accelerator':'gpu', 'devices':args.n_gpus, 'num_nodes':1, 'strategy':'ddp'}
        
    trainer = Trainer(
        max_epochs=args.num_train_epochs,
        log_every_n_steps=args.log_every_n_steps, 
        check_val_every_n_epoch=1, 
        default_root_dir=args.output_dir,
        callbacks=callbacks,
        enable_progress_bar=False, 
        logger=logger, 
        deterministic=True, 
        **device_kwargs, 
    )
    model.training_steps = args.max_steps
    model.validation_steps = args.val_steps
    
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    
  
if __name__ == '__main__':
    parser = get_train_parser()
    parser.add_argument('--dataset', type=str, required=True, help="Choose between 'covidx' and 'rsna'")
    parser.add_argument('--unfreeze_adaptor', action='store_true')
    args = parser.parse_args()

    print('Number of GPUs available:', torch.cuda.device_count())
    main(args)
    