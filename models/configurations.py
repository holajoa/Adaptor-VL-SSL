from mgca.datasets.transforms import DataTransforms
from utils.dataset_utils import AutoEncoderDataTransforms


TEXT_PRETRAINED_AVAILABLE = [
    "bert-base-uncased", 
    "dmis-lab/biobert-v1.1", 
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
    "microsoft/BiomedVLP-CXR-BERT-general", 
    "./weights/ClinicalBERT_checkpoint/ClinicalBERT_pretraining_pytorch_checkpoint", 
]

VISION_PRETRAINED_AVAILABLE = {
    ### pretrained_weight: vision_model_type
    "101-elastic": "ae", 
    "swin_base_patch4_window7_224": "timm",
}

VISION_MODEL_TYPE_2_DATA_TRANSFORM = {
    'ae': AutoEncoderDataTransforms,
    'timm': DataTransforms,
    'huggingface': DataTransforms,
}

VISION_MODEL_TYPE_2_VISION_OUTPUT_DIM = {
    'ae': 512,
    'timm': 1024, 
}