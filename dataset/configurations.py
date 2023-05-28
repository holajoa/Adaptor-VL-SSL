from mgca.datasets.classification_dataset import RSNAImageDataset, COVIDXImageDataset

DATASET_CFG = {
    'rsna': {
        'class':RSNAImageDataset,
        'kwargs':{'phase':'classigication'}, 
    }, 
    'covidx':{
        'class':COVIDXImageDataset, 
        'kwargs':dict(), 
    },
}
