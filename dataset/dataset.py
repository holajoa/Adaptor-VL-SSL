from mgca.datasets.pretrain_dataset import (
    MultimodalPretrainingDataset, 
    multimodal_collate_fn, 
    BASE_DIR, 
)
from mgca.constants import *
from mgca.datasets.utils import get_imgs

from transformers import BertTokenizer
from transformers.tokenization_utils import PreTrainedTokenizerBase
from datasets import Dataset, concatenate_datasets

import torch 

import os
from pathlib import Path
import pickle

from tqdm import tqdm

from pandas import read_csv
import numpy as np

from nltk.tokenize import RegexpTokenizer
import re


class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 split='train', 
                 transform=None, 
                 data_pct=1.0, 
                 imsize=256, 
                 tokenizer=None,
                 max_words=112):
        super().__init__()
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(f"{MIMIC_CXR_DATA_DIR} does not exist!")
        
        self.split = split
        self.transform = transform
        self.imsize = imsize
        self.df = None
        self.max_words = max_words
        
        if isinstance(tokenizer, str):
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        elif isinstance(tokenizer, PreTrainedTokenizerBase):
            self.tokenizer = tokenizer
        
    def load_text_data(self):
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
        total = len(self.df)
        removed_indices = []
        with tqdm(total=total) as pbar:
            for row in self.df.itertuples():
                cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
                path = getattr(row, MIMIC_CXR_PATH_COL)
                
                if cur_split == self.split and path in path2sent:
                    ### Addition: make sure path points to an existing file ==========
                    if not Path(path).is_file():
                        removed_indices.append(row.Index)
                        continue
                    ### End addition =================================================
                    filenames.append(path)
                
                pbar.update(1)
        return filenames, path2sent, removed_indices
    
    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []
        path2sent = {}
        # iterrows is not faster than itertuples ...  but it is ok
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # pick impression, findings, last_paragraph
            captions = ""
            captions += row["impression"]
            captions += " "
            captions += row["findings"]

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:
                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())
                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))

                cnt += len(included_tokens)

            if cnt >= 3:
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2sent[row[MIMIC_CXR_PATH_COL]] = study_sent

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent
    
    def get_caption(self, path):
        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len
    
    def __get__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    

class MultimodalPretrainingDatasetForAdaptor(MultimodalDataset):
    def __init__(self, 
                 split='train', 
                 transform=None, 
                 data_pct=1.0, 
                 imsize=256, 
                 tokenizer=None,
                 max_words=112):
        super().__init__(split=split, transform=transform, data_pct=data_pct, 
                         imsize=imsize, tokenizer=tokenizer, max_words=max_words)
        
        self.df = read_csv(MIMIC_CXR_MASTER_CSV)
        self.df = self.df[self.df[MIMIC_CXR_VIEW_COL].isin(["PA", "AP"])]
        self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
            lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:])))

        # load studies and study to text mapping
        self.filenames, self.path2sent, removed_indices = self.load_text_data()
        self.df.drop(removed_indices, axis=0, inplace=True)  # remove rows where image file path does not exist
        
        self.df = self.df[self.df[MIMIC_CXR_SPLIT_COL] == split]
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)
        self.df.dicom_id.to_csv(f'{MIMIC_CXR_DATA_DIR}/mimic-cxr-2.0.0_idx2dicom_id_{split}.csv')
        
    def __getitem__(self, index):
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        imgs = get_imgs(key, self.imsize, self.transform, multiscale=False)
        return imgs, caps, cap_len, key
    
    def __len__(self):
        return len(self.filenames)

    
class MultimodalPretrainedEmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        text_model_name: str,
        image_model_name: str,
        split: str='train',
        device='cpu',
        num_of_samples=-1, 
        shuffle=False, 
        seed=42, 
    ):
        super().__init__()
        self.text_embeds_raw_dir = f'saved_embeddings/text_embeds/{text_model_name}_{split}.npy'
        self.image_embeds_raw_dir = f'saved_embeddings/image_embeds/{image_model_name}_{split}.npy'
        
        self.image_embeds = np.load(self.image_embeds_raw_dir, mmap_mode='r')
        self.text_embeds = np.load(self.text_embeds_raw_dir, mmap_mode='r')
        
        self.device = torch.device(device)
        if num_of_samples > 0:
            self.indices = self.indices[:num_of_samples]
        else:  ## num_of_samples == -1
            num_of_samples = len(self.image_embeds)
        self.num_of_samples = num_of_samples
        self.indices = np.arange(self.num_of_samples)
        if shuffle and split == 'train':
            np.random.seed(seed)
            np.random.shuffle(self.indices)
        
    def __getitem__(self, idx):
        text_tensor = self.text_embeds[self.indices[idx]]
        image_tensor = self.image_embeds[self.indices[idx]]
        return {'text_embeds_raw':text_tensor, 'image_embeds_raw':image_tensor}

    def __len__(self):
        return len(self.indices)
    
    
# class MultimodalPretrainedEmbeddingsIterableDataset(torch.utils.data.IterableDataset):
#     def __init__(
#         self, 
#         text_embeds_raw_dir: str,
#         image_embeds_raw_dir: str,
#         split: str='train',
#         device='cpu',
#         num_of_samples=-1, 
#         shuffle=True,
#     ):
#         super().__init__()
#         self.text_embeds_raw_dir = os.path.join(text_embeds_raw_dir, split)
#         self.image_embeds_raw_dir = os.path.join(image_embeds_raw_dir, split)
#         self.device = torch.device(device)
#         self.num_of_samples = num_of_samples
#         self.indices = np.arange(self.num_of_samples)
        
#         if shuffle and split == 'train':
#             self.shuffle()
        
#         assert self.text_tensor_names == self.image_tensor_names, "text and image tensor names do not match"
#         self.batch_size = torch.load(os.path.join(self.text_embeds_raw_dir, self.text_tensor_names[0]), 
#                                      map_location='cpu')[0].shape[0]
    
#     def process_single_tensor_file(self, text_tensor, image_tensor):
#         for tt, it in zip(text_tensor, image_tensor):
#             yield {'text_embeds_raw':tt, 'image_embeds_raw':it}
    
#     def shuffle(self):
#         self.indices = torch.randperm(self.num_of_samples)
        
    
#     def __iter__(self):
#         for text_tensor_name, image_tensor_name in zip(self.text_tensor_names, self.image_tensor_names):
#             text_tensor = torch.load(os.path.join(self.text_embeds_raw_dir, text_tensor_name), 
#                                      map_location=self.device)
#             image_tensor = torch.load(os.path.join(self.image_embeds_raw_dir, image_tensor_name), 
#                                       map_location=self.device)
#             if isinstance(image_tensor, dict):  ### For ResNetAE
#                 image_tensor = image_tensor['z']
            
#             yield from self.process_single_tensor_file(text_tensor, image_tensor)
    
#     def __len__(self):
#         return self.num_of_batches * self.batch_size
    

# class MultimodalDatasetForClassification(MultimodalDataset):
#     def __init__(
#         self, 
#         split='train', 
#         transform=None, 
#         data_pct=1.0, 
#         img_type='Frontal', 
#         imsize=256, 
#         tokenizer=None,
#         max_words=112, 
#     ):
#         super().__init__(split=split, transform=transform, data_pct=data_pct, img_type=img_type, 
#                          imsize=imsize, tokenizer=tokenizer, max_words=max_words)
        
#         if not os.path.exists(MIMIC_CXR_DATA_DIR):
#             raise RuntimeError(
#                 "MIMIC CXR data directory %s does not exist!" % MIMIC_CXR_DATA_DIR)

#         # read in csv file
#         if split == "train":
#             self.df = read_csv(MIMIC_CXR_TRAIN_CSV)
#         elif split == "valid":
#             self.df = read_csv(MIMIC_CXR_VALID_CSV)
#         else:
#             self.df = read_csv(MIMIC_CXR_TEST_CSV)

#         # filter image type
#         if img_type != "All":
#             self.df = self.df[self.df[MIMIC_CXR_VIEW_COL].isin(["PA", "AP"])]
        
#         # get a fraction of dataset
#         if data_pct != 1.0 and split == "train":
#             self.df = self.df.sample(frac=data_pct, random_state=42)
            
#         # get path
#         self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
#             lambda x: os.path.join(
#                 MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:]))
#         )

#         # fill na with 0s
#         self.df = self.df.fillna(0)

#         # replace uncertains
#         uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
#         self.df = self.df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)
    
#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, index):
#         row = self.df.iloc[index]

#         # get image
#         img_path = row["Path"]
#         x = get_imgs(img_path, self.imsize, self.transform)

#         # get labels
#         y = list(row[CHEXPERT_COMPETITION_TASKS])
#         y = torch.tensor(y)
        
#         # Get caption
#         key = self.filenames[index]
#         caps, cap_len = self.get_caption(key)

#         # return x, y, img_path
#         return x, caps, cap_len, y, key, img_path


def multimodal_collator(*args, **kwargs):
    d = multimodal_collate_fn(*args, **kwargs)
    d['input_ids'] = d.pop('caption_ids')
    d['pixel_values'] = d.pop('imgs')
    d['return_loss'] = True
    return d
