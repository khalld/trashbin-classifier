import random
import numpy as np
import pandas as pd
from torchvision import transforms
from os.path import join
from PIL import Image
import torch

from libs.TrashbinDataset import TrashbinDataset
from libs.TDContainer import TDContainer
from libs.PretrainedModels import PretrainedModelsCreator, CCAlexNet, CCAlexNet_rg
from libs.Training import AverageValueMeter, trainval_classifier, test_classifier, accuracy_score
from libs.utils import get_model_name, import_dataset

def train(  creator: PretrainedModelsCreator, model_name: str,
            dataset: TDContainer, output_class: int, batch_size: int, num_workers: int, drop_last: bool, # parametri di initialize_dst
            lr: float, epochs: int, momentum: float = 0.99,     # parametri di trainval_classifier
            loaded_model: str='',
            logdir='logs', modeldir='models', train_from_epoch: int=0, save_on_runtime: bool = False, save_each_iter: int=5, resume_global_step_from: int=0) -> None:

    print('**** Instantiating %s' % (model_name))
    creator.initialize_dst(dataset, output_class, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last)

    if len(loaded_model) > 0:
        print('**** Loading model')
        creator.load_model(loaded_model)

    print('**** Starting procedure ***')
    model_finetuned = trainval_classifier(model=creator.model, dst_container=dataset, model_name=model_name, lr=lr, epochs=epochs, momentum=momentum, logdir=logdir, modeldir=modeldir, train_from_epoch=train_from_epoch, save_on_runtime=save_on_runtime, save_each_iter=save_each_iter, logs_txt=True,
                                            resume_global_step_from=resume_global_step_from) ## nuovo parametro preso da tensorboard!!! è il numero di step
    print("**** Start to calculate accuracy ...")
    model_finetuned_predictions_test, dataset_labels_test = test_classifier(model_finetuned, dataset.test_loader)
    print("**** Accuracy of %s %0.2f%%" % (model_name, accuracy_score(dataset_labels_test, model_finetuned_predictions_test)*100) )
    print('**** Ended %s' % (model_name))

if __name__ == "__main__":   
    random.seed(1996)
    np.random.seed(1996)

    # continuo col dataset di 04-train.py
    
    dataset_v3 = import_dataset('dataset', 
        train_transform=transforms.Compose([
            transforms.Resize(256),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.RandomCrop(224),
            transforms.RandomApply(torch.nn.ModuleList([
                transforms.Grayscale(num_output_channels=3), # tutti i modelli richiedono un'immagine a tre livelli
            ]), p=0.3), # effettuo un grayscale con probabilità 0.3
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # default dev and std for pretrained models
        ]),
        test_transform=transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), # crop centrale
            transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN), # NOTA: già su dataset_v2 è stato settato sul train
            transforms.RandomApply(torch.nn.ModuleList([
                transforms.Grayscale(num_output_channels=3), # tutti i modelli richiedono un'immagine a tre livelli
            ]), p=0.2), # effettuo un grayscale con probabilità 0.2
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # default dev and std for pretrained models
        ])
    )

    # effettuo un ulteriore training di 25 epoche senza parametri freezati per migliorare il modello con LR collegato
    # ma aggiornando anche i parametri

    # AlexNet__lr=0.001
    train(creator=CCAlexNet_rg(), model_name=get_model_name(model_name="AlexNet", lr="0.001"), dataset=dataset_v3, output_class=3, batch_size=64, num_workers=2, drop_last=True, lr=0.001, epochs=25, save_each_iter=4,
        loaded_model=join('models/AlexNet__lr=0.001', 'AlexNet__lr=0.001-40.pth'),
        train_from_epoch=40, resume_global_step_from=185455)
    
    # AlexNet__lr=0.0003
    train(creator=CCAlexNet_rg(), model_name=get_model_name(model_name="AlexNet", lr="0.0003"), dataset=dataset_v3, output_class=3, batch_size=64, num_workers=2, drop_last=True, lr=0.0003, epochs=25, save_each_iter=4,
        loaded_model=join('models/AlexNet__lr=0.0003', 'AlexNet__lr=0.0003-40.pth'),
        train_from_epoch=40, resume_global_step_from=185455)

    # 2 dataset

    # AlexNet_2dst__lr=0.001
    train(creator=CCAlexNet_rg(), model_name=get_model_name(model_name="AlexNet_2dst", lr="0.001"), dataset=dataset_v3, output_class=3, batch_size=64, num_workers=2, drop_last=True, lr=0.001, epochs=25, save_each_iter=4,
        loaded_model=join('models/AlexNet_2dst__lr=0.001', 'AlexNet_2dst__lr=0.001-40.pth'),
        train_from_epoch=40, resume_global_step_from=185455)

    # AlexNet_2dst__lr=0.0003
    train(creator=CCAlexNet_rg(), model_name=get_model_name(model_name="AlexNet_2dst", lr="0.0003"), dataset=dataset_v3, output_class=3, batch_size=64, num_workers=2, drop_last=True, lr=0.0003, epochs=25, save_each_iter=4,
        loaded_model=join('models/AlexNet_2dst__lr=0.0003', 'AlexNet_2dst__lr=0.0003-40.pth'),
        train_from_epoch=40, resume_global_step_from=185455)
