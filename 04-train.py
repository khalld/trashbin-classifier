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

def train(  creator: PretrainedModelsCreator, model_name: str, 
            dataset: TDContainer, output_class: int, batch_size: int, num_workers: int, drop_last: bool, # parametri di initialize_dst
            lr: float, epochs: int, momentum: float = 0.99,     # parametri di trainval_classifier
            logdir='logs', modeldir='models', train_from_epoch: int=0, save_on_runtime: bool = False, save_each_iter: int=5) -> None:

    print('*** Instantiating %s' % (model_name))
    creator.initialize_dst(dataset, output_class, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last)
    print("\n")
    print('*** Starting procedure ***')
    model_finetuned = trainval_classifier(model=creator.model, dst_container=dataset, model_name=model_name, lr=lr, epochs=epochs, momentum=momentum, logdir=logdir, modeldir=modeldir, train_from_epoch=train_from_epoch, save_on_runtime=save_on_runtime, save_each_iter=save_each_iter)
    print("\n")
    print("**** Start to calculate accuracy ...")
    model_finetuned_predictions_test, dataset_labels_test = test_classifier(model_finetuned, dataset.test_loader)
    print("\n")
    print("**** Accuracy of %s %0.2f%%" % (model_name, accuracy_score(dataset_labels_test, model_finetuned_predictions_test)*100) )
    print('**** Ended %s' % (model_name))

# def load_and_train(creator: PretrainedModelsCreator, path_model: str, dataset: TDContainer, output_class: int,
#                 model_name: str, batch_size: int, num_workers: int, 
#                 drop_last: bool, lr: float, epochs: int, train_from_epoch: int ) -> None:

#     print('Instantiating %s' % (model_name))
#     creator.initialize_dst(dataset, output_class, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last)

#     print('Loading model from %s' % (model_name))
#     creator.load_model(path_model)

#     print('*** Starting procedure ***')
#     model_finetuned = trainval_classifier(model=creator.model, dst_container=dataset, model_name=model_name,
#                                         lr=lr, epochs=epochs, train_from_epoch=train_from_epoch,
#                                         logdir=GENERAL_PATHS['logs'], model_dir=GENERAL_PATHS['models'], save_each_iter=5)
#     print("**** Start to calculate accuracy ...")
#     model_finetuned_predictions_test, dataset_labels_test = test_classifier(model_finetuned, dataset.test_loader)
#     print("**** Accuracy of %s %0.2f%%" % (model_name, accuracy_score(dataset_labels_test, model_finetuned_predictions_test)*100) )

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

    # check corretto caricamento dataset
    # dataset_v1.show_info()

    # test 1 epoca con VGG16
    
    train(creator=CCMobileNetV2(), model_name=get_model_name(model_name="MobileNetV2", version="1", lr=0.1), dataset=dataset_v1, output_class=3, batch_size=64, num_workers=2, drop_last=False, lr=0.01, epochs=1, save_each_iter=1 )