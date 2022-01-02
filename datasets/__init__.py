from datasets.settings import *

if DATASET_TYPE == "IMAGENET1K":
    from datasets.imagenet1k import train_data, valid_data, train_loader, valid_loader
elif DATASET_TYPE == "CIFAR10":
    from datasets.cifar10 import train_data, valid_data, train_loader, valid_loader
else:
    raise Exception("Wrong DATASET_TYPE!")
