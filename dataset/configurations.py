from mgca.datasets.classification_dataset import RSNAImageDataset, COVIDXImageDataset

DATASET_CFG = {
    'rsna': {
        'class':RSNAImageDataset,
        'kwargs':{'phase':'classigication'}, 
        'num_classes':1, 
    }, 
    'covidx':{
        'class':COVIDXImageDataset, 
        'kwargs':dict(), 
        'num_classes':3, 
    },
}
