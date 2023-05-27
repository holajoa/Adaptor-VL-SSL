import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger

from mgca.datasets.classification_dataset import RSNAImageDataset, COVIDXImageDataset
from mgca.datasets.transforms import DataTransforms

from models.adaptor import StreamingProgressBar
from models.pipeline import AdaptorPipelineWithClassificationHead
from models.configurations import TEXT_PRETRAINED, VISION_PRETRAINED
from utils.model_utils import get_newest_ckpt
from utils.dataset_utils import torch2huggingface_dataset, get_dataloader
from dataset.dataset import clf_collator
from utils.args import get_train_parser

from math import ceil
import logging


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
    
    seed_everything(42)
    
    from time import gmtime, strftime
    log_fn = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    logger = CSVLogger(args.output_dir)
    
    trainer = Trainer(
        accelerator="gpu", 
        devices=args.n_gpu, 
        strategy="ddp", 
        # accelerator="cpu",
        max_epochs=args.num_train_epochs,
        # max_steps=args.max_steps,
        log_every_n_steps=200, 
        check_val_every_n_epoch=1, 
        default_root_dir=args.output_dir,
        callbacks=[StreamingProgressBar(total=args.max_steps//args.num_train_epochs, 
                                        val_total=args.val_steps)],
        enable_progress_bar=False, 
        logger=logger, 
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    
  
if __name__ == '__main__':
    parser = get_train_parser()
    parser.add_argument('--dataset', type=str, required=True, help="Choose between 'covidx' and 'rsna'")
    args = parser.parse_args()

    # from time import gmtime, strftime

    # log_fn = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    # logging.basicConfig(filename=f'logs/clf_finetune_{log_fn}.log', encoding='utf-8', level=logging.INFO)

    # num_of_gpus = torch.cuda.device_count()
    # logging.info(f"Number of available GPUs = {num_of_gpus}: "
    #             f"{', '.join([torch.cuda.get_device_properties(i).name for i in range(num_of_gpus)])}.")
    
    main(args)
    