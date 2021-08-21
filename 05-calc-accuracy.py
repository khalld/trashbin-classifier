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


def calc_accuracy(  creator: PretrainedModelsCreator, model_name: str,
            dataset: TDContainer, output_class: int, batch_size: int, num_workers: int, drop_last: bool, # parametri di initialize_dst
            loaded_model: str='') -> None:

    creator.initialize_dst(dataset, output_class, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last)
    creator.load_model(loaded_model)
    
    print('***** Calculating accuracy of %s *****' % (model_name) )
    model_finetuned_predictions_test, dataset_labels_test = test_classifier(creator.model, dataset.test_loader)
    print("**** Current accuracy of %s %0.2f%%" % (model_name, accuracy_score(dataset_labels_test, model_finetuned_predictions_test)*100) )

if __name__ == "__main__":   
    random.seed(1996)
    np.random.seed(1996)

    dataset_v1 = import_dataset('dataset', 
        train_transform=transforms.Compose([
            transforms.Resize(256),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomPerspective(p=0.3),
            transforms.RandomVerticalFlip(p=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # default dev and std for pretrained models
        ]),
        test_transform=transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), # crop centrale
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # default dev and std for pretrained models
        ])
    )

    # Dopo aver ottenuto un allenato un totale di 4 modelli per 40 epoche,
    # calcolo la rispettiva accuracy di ogni modello dopo 40 iterazioni e vedo qual è la migliore

    # AlexNet__lr=0.001
    calc_accuracy(creator=CCAlexNet(), model_name='AlexNet__lr=0.001',
        dataset=dataset_v1, output_class=3, batch_size=64, num_workers=2, drop_last=True,
        loaded_model=join('models/AlexNet__lr=0.001', 'AlexNet__lr=0.001-40.pth'))

    # AlexNet__lr=0.0003
    calc_accuracy(creator=CCAlexNet(), model_name='AlexNet__lr=0.0003',
        dataset=dataset_v1, output_class=3, batch_size=64, num_workers=2, drop_last=True,
        loaded_model=join('models/AlexNet__lr=0.0003', 'AlexNet__lr=0.0003-40.pth'))


    # AlexNet_2dst__lr=0.001
    calc_accuracy(creator=CCAlexNet(), model_name='AlexNet_2dst__lr=0.001',
            dataset=dataset_v1, output_class=3, batch_size=64, num_workers=2, drop_last=True,
            loaded_model=join('models/AlexNet_2dst__lr=0.001', 'AlexNet_2dst__lr=0.001-40.pth'))

    # AlexNet_2dst__lr=0.0003
    calc_accuracy(creator=CCAlexNet(), model_name='AlexNet_2dst__lr=0.0003',
            dataset=dataset_v1, output_class=3, batch_size=64, num_workers=2, drop_last=True,
            loaded_model=join('models/AlexNet_2dst__lr=0.0003', 'AlexNet_2dst__lr=0.0003-40.pth'))

    # con il modello scelto

    # Current accuracy of AlexNet__lr=0.001 97.28%
    # **** Current accuracy of AlexNet__lr=0.0003 96.93%
    # **** Current accuracy of AlexNet_2dst__lr=0.001 95.36%
    # **** Current accuracy of AlexNet_2dst__lr=0.0003 96.52%


    # arriva a un tot di 50 epoche trained
    # abilita i pesi freezati e fallo per altre 50 epoche