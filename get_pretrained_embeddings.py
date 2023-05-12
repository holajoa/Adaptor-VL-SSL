import torch 
import torch.nn as nn

from transformers import BertModel

from models.configurations import (VISION_MODEL_TYPE_2_DATA_TRANSFORM, 
                                   VISION_MODEL_TYPE_2_VISION_OUTPUT_DIM)
from utils.utils import get_text_embeds_raw, get_image_embeds_raw

from utils.dataset_utils import get_dataloader
from utils.dataset_utils import pickle_dataset
from utils.model_utils import load_vision_model

import logging

from dataset.dataset import multimodal_collator
import logging

import argparse


# SAVED_EMBEDDINGS_DIR = '/vol/bitbucket/jq619/individual-project/saved_embeddings'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--vision_pretrained', type=str, help='Choose from [101-elastic, swin_base_patch4_window7_224]')
parser.add_argument('--vision_model_type', type=str, help='Choose from [timm, ae, huggingface]')
parser.add_argument('--text_pretrained', type=str, 
                    help='Choose from [bert-base-uncased, dmis-lab/biobert-v1.1, '
                    'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext, '
                    'microsoft/BiomedVLP-CXR-BERT-general, '
                    './weights/ClinicalBERT_checkpoint/ClinicalBERT_pretraining_pytorch_checkpoint]')

parser.add_argument('--text_embeds_raw_dir', type=str, help='path to raw text embeddings')
parser.add_argument('--image_embeds_raw_dir', type=str, help='path to raw image embeddings')
# parser.add_argument('--num_of_batches', type=int, default=100, help='number of batches to use for training')

parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--force_rebuild_dataset', action='store_true', help='Whether to force rebuild dataset, if not can load pickled file if available')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--data_pct', type=float, default=0.01, help='percentage of data to use. If setting 1.0, then use all data with no shuffling')
parser.add_argument('--crop_size', type=int, default=224)

# parser.add_argument('--num_hidden_layers', type=int, default=1, help='number of transformer layers to use in adaptor')
# parser.add_argument('--projection_dim', type=int, default=768, help='dimension of projection head')

# parser.add_argument('--lr', type=float, default=1e-4)
# parser.add_argument('--num_train_epochs', type=int, default=1)
parser.add_argument('--seed', type=int, default=1117)
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

args = parser.parse_args()

torch.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')
print(f'Using device: {device}')

do_text = args.text_pretrained is not None
do_vision = args.vision_pretrained is not None and args.vision_model_type is not None

# Load pretrained models
if args.vision_model_type is None or args.vision_pretrained is None:
    args.vision_model_type = 'ae'
    args.vision_pretrained = '101-elastic'
vision_model = load_vision_model(args.vision_model_type, args.vision_pretrained, retain_head=False)
vision_model.to(device)


if args.text_pretrained is None:
    args.text_pretrained = "dmis-lab/biobert-v1.1"
text_model = BertModel.from_pretrained(args.text_pretrained)
text_model.to(device)

### Load dataset
data_transforms = VISION_MODEL_TYPE_2_DATA_TRANSFORM[args.vision_model_type]

train_dataset_pkl = f'saved_datasets/train_dataset_{args.vision_model_type}.pkl'
val_dataset_pkl = f'saved_datasets/val_dataset_{args.vision_model_type}.pkl'
test_dataset_pkl = f'saved_datasets/test_dataset_{args.vision_model_type}.pkl'

train_dataset = pickle_dataset(
    train_dataset_pkl, 
    split='train', 
    transform=data_transforms(True, args.crop_size), 
    data_pct=args.data_pct, 
    force_rebuild=args.force_rebuild_dataset, 
)
val_dataset = pickle_dataset(
    val_dataset_pkl,
    split='valid',
    transform=data_transforms(False, args.crop_size),
    data_pct=args.data_pct, 
    force_rebuild=args.force_rebuild_dataset, 
)

test_dataset = pickle_dataset(
    test_dataset_pkl,
    split='test',
    transform=data_transforms(False, args.crop_size),
    data_pct=args.data_pct, 
    force_rebuild=args.force_rebuild_dataset, 
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
test_dataloader = get_dataloader(
    test_dataset, 
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=multimodal_collator,
)


for split, dataloader in zip(['train', 'valid', 'test'], 
                             [train_dataloader, val_dataloader, test_dataloader]):
    if do_vision:
        logging.info(f'Getting vision embeddings for {split} split')
        get_image_embeds_raw(
            dataloader,
            vision_model=vision_model,
            vision_model_type=args.vision_model_type,  
            save_path=args.image_embeds_raw_dir,
            model_name=args.vision_pretrained,
            batch_size=args.batch_size,
            embedding_dim=VISION_MODEL_TYPE_2_VISION_OUTPUT_DIM[args.vision_model_type],
            split=split,
            device=device,
        )
    if do_text:
        logging.info(f'Getting text embeddings for {split} split')
        get_text_embeds_raw(
            dataloader,
            text_model=text_model,
            save_path=args.text_embeds_raw_dir,
            model_name=args.text_pretrained,
            batch_size=args.batch_size,
            split=split,
            device=device,
        )
    