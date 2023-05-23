import torch 

from transformers import AutoTokenizer
from transformers import BertModel
from pytorch_lightning import Trainer, seed_everything

from models.adaptor import Adaptor
from models.configurations import TEXT_PRETRAINED, VISION_PRETRAINED
from utils.dataset_utils import pickle_dataset, get_dataloader
from utils.model_utils import load_vision_model
from dataset.dataset import multimodal_collator

import argparse 
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
    data_transforms = vision_model_config['data_transform']
    
    args.text_pretrained = TEXT_PRETRAINED[args.text_model]

    ### Load vision model
    vision_model = load_vision_model(args.vision_model_type, args.vision_pretrained)

    ### Load text model
    text_model = BertModel.from_pretrained(args.text_pretrained)
    tokenizer = AutoTokenizer.from_pretrained(args.text_pretrained)

    ### Define model
    model = Adaptor(
        text_model=text_model,
        vision_model=vision_model,
        vision_model_type=args.vision_model_type, 
        vision_output_dim=args.vision_output_dim,
        projection_dim=args.projection_dim,
        num_hidden_layers=args.num_hidden_layers, 
        lr=args.lr, 
    )

    ### Load dataset
    postfix = '_ae' if args.vision_model == 'ae' else ''
    train_dataset_pkl = f'saved_datasets/train_dataset_{args.text_model}{postfix}.pkl'
    val_dataset_pkl = f'saved_datasets/val_dataset_{args.text_model}{postfix}.pkl'


    train_dataset = pickle_dataset(
        train_dataset_pkl, 
        split='train', 
        transform=data_transforms(True, args.crop_size), 
        data_pct=args.data_pct, 
        force_rebuild=args.force_rebuild_dataset, 
        tokenizer=tokenizer,
    )
    val_dataset = pickle_dataset(
        val_dataset_pkl,
        split='valid',
        transform=data_transforms(False, args.crop_size),
        data_pct=args.data_pct, 
        force_rebuild=args.force_rebuild_dataset, 
        tokenizer=tokenizer,
    )

    # Get dataloaders
    train_dataloader = get_dataloader(
        train_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=multimodal_collator,
    )
    val_dataloader = get_dataloader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=multimodal_collator,
    )
    
    ### Training
    seed_everything(args.seed)
    trainer = Trainer(
        accelerator="gpu", 
        devices=args.n_gpu, 
        strategy="ddp" , 
        # accelerator="cpu",
        max_epochs=args.num_train_epochs,
        log_every_n_steps=20, 
        val_check_interval=50, 
        default_root_dir=args.output_dir,
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vision_model', type=str, help='Choose from [resnet-ae, swin-base]')
    parser.add_argument('--text_model', type=str, 
                        help='Choose from [bert, biobert, pubmedbert, cxrbert, clinicalbert]')
    
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--force_rebuild_dataset', action='store_true', help='Whether to force rebuild dataset, if not can load pickled file if available')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--data_pct', type=float, default=1.0, help='percentage of data to use')
    parser.add_argument('--crop_size', type=int, default=224)

    parser.add_argument('--num_hidden_layers', type=int, default=1, help='number of transformer layers to use in adaptor')
    parser.add_argument('--projection_dim', type=int, default=768, help='dimension of projection head')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_train_epochs', type=int, default=1)

    parser.add_argument('--output_dir', type=str, default='./results', help='path to save model')
    
    parser.add_argument('--n_gpu', type=int, default=2, help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=1117)
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

    args = parser.parse_args()


    from time import gmtime, strftime

    log_fn = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    logging.basicConfig(filename=f'logs/train_{log_fn}.log', encoding='utf-8', level=logging.INFO)

    num_of_gpus = torch.cuda.device_count()
    logging.info(f"Number of available GPUs = {num_of_gpus}: "
                f"{', '.join([torch.cuda.get_device_properties(i).name for i in range(num_of_gpus)])}.")
    
    main(args)
    