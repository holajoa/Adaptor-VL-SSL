from mgca.datasets.pretrain_dataset import (
    MultimodalPretrainingDataset, 
    multimodal_collate_fn, 
    BASE_DIR, 
)
from mgca.datasets.transforms import DataTransforms
from mgca.constants import *

from transformers import BertTokenizer

import torch 
import torch.nn as nn
import os
from pathlib import Path
import pickle

from tqdm import tqdm


class MultimodalPretrainingDatasetForAdaptor(MultimodalPretrainingDataset):
    def __init__(self, split="train", transform=None, data_pct=1.0,
                 imsize=256, max_words=128, sent_num=3, tokenizer=None):
        super().__init__(split=split, transform=transform, data_pct=data_pct, 
                         imsize=imsize, max_words=max_words, sent_num=sent_num)
        if isinstance(tokenizer, str):
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        elif isinstance(tokenizer, nn.Module):
            self.tokenizer = tokenizer
    
    def load_text_data(self, split):
        # get study to captions mapping
        # TODO: check this
        filepath = os.path.join(BASE_DIR, "../../data/captions.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)
    
        # filter studies to use for current split
        filenames = []
        for row in tqdm(self.df.itertuples()):
            cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
            path = getattr(row, MIMIC_CXR_PATH_COL)
            
            ### Addition: make sure path points to an existing file 
            if not Path(path).is_file():
                continue
            
            if cur_split == split and path in path2sent:
                filenames.append(path)
                
        return filenames, path2sent

def multimodal_collator(*args, **kwargs):
    d = multimodal_collate_fn(*args, **kwargs)
    d['input_ids'] = d.pop('caption_ids')
    d['pixel_values'] = d.pop('imgs')
    return d