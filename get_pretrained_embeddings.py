import torch
import torch.nn as nn

from transformers import BertModel, AutoTokenizer

from models.configurations import VISION_PRETRAINED, TEXT_PRETRAINED
from utils.utils import get_text_embeds_raw, get_image_embeds_raw

from utils.dataset_utils import get_dataloader
from utils.dataset_utils import pickle_dataset
from utils.model_utils import load_vision_model

import logging

from dataset.dataset import multimodal_collator
import logging

import os
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--vision_model", type=str, help="Choose from [resnet-ae, swin-base]"
)
parser.add_argument(
    "--text_model",
    type=str,
    help="Choose from [bert, biobert, pubmedbert, cxrbert, clinicalbert]",
)

parser.add_argument(
    "--text_embeds_raw_dir",
    type=str,
    default="/vol/bitbucket/jq619/adaptor-thesis/saved_embeddings/text_embeds",
    help="path to raw text embeddings",
)
parser.add_argument(
    "--image_embeds_raw_dir",
    type=str,
    default="/vol/bitbucket/jq619/adaptor-thesis/saved_embeddings/image_embeds",
    help="path to raw image embeddings",
)
# parser.add_argument('--num_of_batches', type=int, default=100, help='number of batches to use for training')

parser.add_argument("--batch_size", type=int, default=32)

parser.add_argument(
    "--force_rebuild_dataset",
    action="store_true",
    help="Whether to force rebuild dataset, if not can load pickled file if available",
)
parser.add_argument("--cpu", action="store_true", help="Whether to run on cpu")

parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument(
    "--data_pct",
    type=float,
    default=0.01,
    help="percentage of data to use. If setting 1.0, then use all data with no shuffling",
)
parser.add_argument("--crop_size", type=int, default=224)

parser.add_argument(
    "--full", action="store_true", help="Compute global and local vision features."
)

# parser.add_argument('--num_hidden_layers', type=int, default=1, help='number of transformer layers to use in adaptor')
# parser.add_argument('--projection_dim', type=int, default=768, help='dimension of projection head')

# parser.add_argument('--lr', type=float, default=1e-4)
# parser.add_argument('--num_train_epochs', type=int, default=1)
parser.add_argument("--seed", type=int, default=1117)
parser.add_argument(
    "--local_rank", default=-1, type=int, help="node rank for distributed training"
)

args = parser.parse_args()

torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
print(f"Using device: {device}")

do_text = args.text_model is not None
do_vision = args.vision_model is not None


if do_vision:
    if args.vision_model not in VISION_PRETRAINED.keys():
        raise ValueError(
            f"Vision model {args.vision_model} not available."
            f"Choose from {list(VISION_PRETRAINED.keys())}"
        )
else:
    args.vision_model = "swin-base"

if do_text:
    if args.text_model not in TEXT_PRETRAINED.keys():
        raise ValueError(
            f"Text model {args.text_model} not available."
            f"Choose from {list(TEXT_PRETRAINED.keys())}"
        )
else:
    args.text_model = "biobert"

vision_model_config = VISION_PRETRAINED[args.vision_model]
args.vision_pretrained = vision_model_config["pretrained_weight"]
args.vision_model_type = vision_model_config["vision_model_type"]
args.vision_output_dim = vision_model_config["vision_output_dim"]
data_transforms = vision_model_config["data_transform"]

args.text_pretrained = TEXT_PRETRAINED[args.text_model]

# Load pretrained models
vision_model = load_vision_model(args.vision_model_type, args.vision_pretrained)
vision_model.to(device)

### Load text model
text_model = BertModel.from_pretrained(args.text_pretrained)
tokenizer = AutoTokenizer.from_pretrained(args.text_pretrained)
text_model.to(device)

### Load dataset
postfix = "_ae" if args.vision_model_type == "ae" else ""
train_dataset_pkl = f"saved_datasets/train_dataset_{args.text_model}{postfix}.pkl"
val_dataset_pkl = f"saved_datasets/val_dataset_{args.text_model}{postfix}.pkl"

train_dataset = pickle_dataset(
    train_dataset_pkl,
    split="train",
    transform=data_transforms(True, args.crop_size),
    data_pct=args.data_pct,
    force_rebuild=args.force_rebuild_dataset,
    validate_path=args.force_rebuild_dataset,
    tokenizer=tokenizer,
)
val_dataset = pickle_dataset(
    val_dataset_pkl,
    split="valid",
    transform=data_transforms(False, args.crop_size),
    data_pct=args.data_pct,
    force_rebuild=args.force_rebuild_dataset,
    validate_path=args.force_rebuild_dataset,
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

os.makedirs(args.image_embeds_raw_dir, exist_ok=True)
os.makedirs(args.text_embeds_raw_dir, exist_ok=True)

for split, dataloader in zip(["train", "valid"], [train_dataloader, val_dataloader]):
    if do_text:
        logging.info(f"Getting text embeddings for {split} split")
        get_text_embeds_raw(
            dataloader,
            text_model=text_model,
            save_path=args.text_embeds_raw_dir,
            model_name=args.text_model,
            batch_size=args.batch_size,
            split=split,
            device=device,
        )
    if do_vision:
        logging.info(f"Getting vision embeddings for {split} split")
        get_image_embeds_raw(
            dataloader,
            vision_model=vision_model,
            vision_model_type=args.vision_model_type,
            save_path=args.image_embeds_raw_dir,
            model_name=args.vision_model,
            batch_size=args.batch_size,
            embedding_dim=args.vision_output_dim,
            split=split,
            device=device,
            full=args.full,
        )
