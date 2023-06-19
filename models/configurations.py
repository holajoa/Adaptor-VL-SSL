from utils.dataset_utils import DataTransforms, AutoEncoderDataTransforms


CACHE_DIR = '/vol/bitbucket/jq619/.cache/huggingface/'

TEXT_PRETRAINED = {
    'bert': 'bert-base-uncased',
    'biobert': 'dmis-lab/biobert-v1.1',
    'pubmedbert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    'cxrbert': 'microsoft/BiomedVLP-CXR-BERT-general',
    'clinicalbert': 'weights/ClinicalBERT_checkpoint/ClinicalBERT_pretraining_pytorch_checkpoint',
}

VISION_PRETRAINED = {
    'resnet-ae':{
        'pretrained_weight': '101-elastic',
        'vision_model_type': 'ae', 
        'data_transform': AutoEncoderDataTransforms,
        'vision_output_dim': 512,
    }, 
    'swin-base':{
        'pretrained_weight': 'swin_base_patch4_window7_224',
        'vision_model_type': 'timm',
        'data_transform': DataTransforms,
        'vision_output_dim': 1024,
    }, 
    'dinov2-s':{
        'pretrained_weight': 'facebookresearch/dinov2/dinov2_vits14',
        'vision_model_type': 'hub',
        'data_transform': DataTransforms,
        'vision_output_dim': 384,
    }, 
    'dinov2-b':{
        'pretrained_weight': 'facebookresearch/dinov2/dinov2_vitb14',
        'vision_model_type': 'hub',
        'data_transform': DataTransforms,
        'vision_output_dim': 768,
    }, 
}
