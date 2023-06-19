from mgca.datasets.classification_dataset import RSNAImageDataset, COVIDXImageDataset
from mgca.datasets.segmentation_dataset import RSNASegmentDataset, SIIMImageDataset

DATASET_CFG = {
    'clf':{
        'rsna': {
            'class':RSNAImageDataset,
            'kwargs':{'phase':'classification'}, 
            'num_classes':1, 
            'binary':True,
            'multilabel':True,
        }, 
        'covidx':{
            'class':COVIDXImageDataset, 
            'kwargs':dict(), 
            'num_classes':3, 
            'binary':False,
            'multilabel':False,
        },
    }, 
    'seg':{
        'rsna':{
            "class": RSNASegmentDataset,
            "kwargs": dict(),
        }, 
        'siim':{
            "class":SIIMImageDataset, 
            "kwargs":{"phase":"segmentation"},
        },
    }
}
