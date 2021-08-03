import random
import numpy as np
import pandas as pd
from torchvision import transforms
from os.path import join
from PIL import Image
import torch

from libs.TrashbinDataset import TrashbinDataset
from libs.TDContainer import TDContainer
from libs.PretrainedModels import PretrainedModelsCreator, CCAlexNet, CCVgg16, CCMobileNetV2
from libs.Training import AverageValueMeter, trainval_classifier, test_classifier, accuracy_score
from libs.utils import get_model_name, import_dataset

def test_model(  creator: PretrainedModelsCreator) -> None: #, model_name: str, 
            # dataset: TDContainer, output_class: int, batch_size: int, num_workers: int, drop_last: bool, # parametri di initialize_dst
            # lr: float, epochs: int, momentum: float = 0.99,     # parametri di trainval_classifier
            # logdir='logs', modeldir='models', train_from_epoch: int=0, save_on_runtime: bool = False, save_each_iter: int=5) -> None:

    creator.initialize_model(3)
    #creator.get_info()
    print(creator.model)


if __name__ == "__main__":   
    random.seed(1996)
    np.random.seed(1996)
    
    test_model(CCAlexNet())