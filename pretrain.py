import torch
import torch.nn as nn

from typing import List, Union, Tuple, Dict, Optional

from dataset.dataset import MultimodalPretrainedEmbeddingsIterableDataset

from models.adaptor import Adaptor, StreamingProgressBar
from models.configurations import (
    TEXT_PRETRAINED_AVAILABLE,
    VISION_PRETRAINED_AVAILABLE,
    VISION_MODEL_TYPE_2_DATA_TRANSFORM,
    VISION_MODEL_TYPE_2_VISION_OUTPUT_DIM, 
)
from utils.model_utils import load_vision_model
from utils.dataset_utils import torch2huggingface_dataset, get_dataloader
from transformers import BertModel

from pytorch_lightning import Trainer, seed_everything

from dataset.dataset import multimodal_collator
import argparse 
import logging


def main(args):
    ### Load vision model (not used in training actually, just for model definition)
    if args.vision_pretrained in VISION_PRETRAINED_AVAILABLE.keys():
        assert VISION_PRETRAINED_AVAILABLE[args.vision_pretrained] == args.vision_model_type, \
            'Vision model type does not match pretrained model'
    vision_model = load_vision_model(args.vision_model_type, args.vision_pretrained)

    ### Load text model (not used in training actually, just for model definition)
    text_model = BertModel.from_pretrained(args.text_pretrained)

    ### Define model
    add_cls_token = args.vision_model_type == 'ae'
    vision_output_dim = VISION_MODEL_TYPE_2_VISION_OUTPUT_DIM[args.vision_model_type]
    model = Adaptor(
        text_model=text_model,
        vision_model=vision_model,
        vision_model_type=args.vision_model_type, 
        vision_output_dim=vision_output_dim,
        projection_dim=args.projection_dim,
        num_hidden_layers=args.num_hidden_layers, 
        add_cls_token=add_cls_token,
        lr=args.lr, 
    )

    ### Load dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_device = 'cpu' if args.num_workers > 0 else device
    train_dataset = MultimodalPretrainedEmbeddingsIterableDataset(args.text_embeds_raw_dir, args.image_embeds_raw_dir, 
                                                                split='train', num_of_batches=args.num_of_batches, 
                                                                device=dataset_device)
    args.max_steps = len(train_dataset) // args.batch_size
    print(f'Length of train dataset: {len(train_dataset)}')
    print(f'Number of steps: {args.max_steps}')

    train_dataset = torch2huggingface_dataset(train_dataset, streaming=True)
    train_dataset.with_format('torch')

    val_dataset = MultimodalPretrainedEmbeddingsIterableDataset(args.text_embeds_raw_dir, args.image_embeds_raw_dir, 
                                                                split='valid', num_of_batches=args.num_of_batches, 
                                                                device=dataset_device)
    val_dataset = torch2huggingface_dataset(val_dataset, streaming=True)
    val_dataset.with_format('torch')

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
    seed_everything(args.seed)
    trainer = Trainer(
        accelerator="gpu", 
        devices=args.n_gpu, 
        strategy="ddp" , 
        # accelerator="cpu",
        max_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        log_every_n_steps=20, 
        val_check_interval=50, 
        default_root_dir=args.output_dir,
        callbacks=[StreamingProgressBar(total=args.max_steps)],
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
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
    parser.add_argument('--num_of_batches', type=int, default=100, help='number of batches to use for training')

    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--force_rebuild_dataset', action='store_true', help='Whether to force rebuild dataset, if not can load pickled file if available')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--data_pct', type=float, default=0.01, help='percentage of data to use')
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
    logging.basicConfig(filename=f'logs/train_from_embeds{log_fn}.log', encoding='utf-8', level=logging.INFO)

    num_of_gpus = torch.cuda.device_count()
    logging.info(f"Number of available GPUs = {num_of_gpus}: "
                f"{', '.join([torch.cuda.get_device_properties(i).name for i in range(num_of_gpus)])}.")
    
    main(args)
    