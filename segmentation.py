import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import pytorch_lightning.callbacks as cb

from models.configurations import TEXT_PRETRAINED, VISION_PRETRAINED
from models.adaptor import Adaptor
from models.segmenter import AdaptorSegmenter
from models.seg_models import ResNetAEUNet, DINOv2Segmenter
from utils.model_utils import get_newest_ckpt, StreamingProgressBar
from dataset.dataset import seg_collator
from dataset.configurations import DATASET_CFG  
from dataset.data_module import AdaptorDataModule
from utils.args import get_train_parser

from math import ceil
import wandb

import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import pytorch_lightning.callbacks as cb

from models.configurations import TEXT_PRETRAINED, VISION_PRETRAINED
from models.finetuner import AdaptorFinetuner
from models.adaptor import Adaptor
from utils.model_utils import get_newest_ckpt, StreamingProgressBar
from dataset.dataset import seg_collator
from dataset.configurations import DATASET_CFG  
from dataset.data_module import AdaptorDataModule
from utils.args import get_train_parser
from utils.model_utils import load_vision_model

from math import ceil
import wandb

import datetime
from dateutil import tz


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
    
    dataset_cfg = DATASET_CFG['seg'][args.dataset]
    dataset_class = dataset_cfg['class']
    dataset_kwargs = dataset_cfg['kwargs']
    if args.vision_model == 'resnet-ae':
        dataset_kwargs['grayscale'] = True

    data_module = AdaptorDataModule(
        dataset=dataset_class, 
        collate_fn=seg_collator, 
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

    vision_model_config = VISION_PRETRAINED[args.vision_model]
    vision_pretrained = vision_model_config['pretrained_weight']
    vision_model_type = vision_model_config['vision_model_type']
    
    adaptor_ckpt = get_newest_ckpt(args.vision_model, args.text_model, wandb=args.wandb)
    adaptor = Adaptor.load_from_checkpoint(adaptor_ckpt)
    
    if args.vision_model == 'resnet-ae':
        seg_model = ResNetAEUNet(
            adaptor=adaptor,
            pretrained=False, 
            out_channels=1, 
        )
    
    elif args.vision_model.startswith('dinov2-'):
        backbone = load_vision_model(
            vision_model_type=vision_model_type, 
            vision_pretrained=vision_pretrained,
            retain_head=False, 
        )
        seg_model = DINOv2Segmenter(
            backbone=backbone,
            adaptor=adaptor,
            hidden_dim=adaptor.projection_dim,
            out_channels=1,
            features=[512, 256, 128, 64], 
        )
    else:
        raise NotImplementedError
    
    model = AdaptorSegmenter(
       seg_model=seg_model,
       adaptor=adaptor,
       learning_rate=args.lr,
       weight_decay=args.weight_decay,      
    )
    
    seed_everything(args.seed, workers=True)
    
    callbacks = [
        StreamingProgressBar(total=data_module.train_steps, val_total=data_module.val_steps), 
    ]
    
    if args.wandb:
        wandb.login(key='b0236e7bef7b6a3789ca4f305406ab358812da3d')
        # now = datetime.datetime.now(tz.tzlocal())
        # extension = now.strftime("%Y_%m_%d_%H_%M_%S")
        logger = WandbLogger(
            project=f"adaptor_segmentation", 
            save_dir=args.output_dir, 
            job_type="train", 
            name=f"{args.vision_model}_{args.text_model}_{args.dataset}_{args.data_pct}",
        )
        logger.watch(model, log_freq=max(100, args.log_every_n_steps))
        logger.log_hyperparams(vars(args))
        experiment_dir = logger.experiment.dir
        callbacks += [
            cb.LearningRateMonitor(logging_interval='step'), 
            # cb.ModelCheckpoint(monitor=f"train_{model.metric_name}_step", mode="max"), 
            cb.ModelCheckpoint(monitor=f"val_{model.metric_name}", mode="max"), 
            cb.ModelCheckpoint(monitor="val_loss", dirpath=args.output_dir, save_last=True, mode="min", save_top_k=3),
            cb.EarlyStopping(monitor=f"val_{model.metric_name}", min_delta=0., patience=10, verbose=False, mode="min")
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
    trainer.test(model, datamodule=data_module, ckpt_path='best')
    
  
if __name__ == '__main__':
    parser = get_train_parser()
    parser.add_argument('--dataset', type=str, required=True, help="Choose between 'covidx' and 'rsna'")
    # parser.add_argument('--unfreeze_adaptor', action='store_true')
    # parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dimension of the classification head")
    # parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate of the classification head")
    args = parser.parse_args()

    print('Number of GPUs available:', torch.cuda.device_count())
    main(args)
    
    