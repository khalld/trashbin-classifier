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
            loaded_model: str='',
            logdir='logs', modeldir='models', train_from_epoch: int=0, save_on_runtime: bool = False, save_each_iter: int=5, resume_global_step_from: int=0) -> None:

    print('**** Instantiating %s' % (model_name))
    creator.initialize_dst(dataset, output_class, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last)

    if len(loaded_model) >0:
        print('**** Loading model')
        creator.load_model(loaded_model)
        
        # da testare il funzionamento:
        print('***** Calculating current accuracy *****')
        model_finetuned_predictions_test, dataset_labels_test = test_classifier(creator.model, dataset.test_loader)
        print("**** Current accuracy of %s %0.2f%%" % (model_name, accuracy_score(dataset_labels_test, model_finetuned_predictions_test)*100) )


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


    # descrivi nella relazione le 3 fasi di questi 3 training

    ## scelta del modello migliore tra gli zoo

    # tutti i modelli prendono in input 224   

    # dataset_v1 = import_dataset('dataset', 
    #     train_transform=transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    #         transforms.RandomCrop(224),
    #         transforms.RandomHorizontalFlip(p=0.4),
    #         transforms.RandomPerspective(p=0.3),
    #         transforms.RandomVerticalFlip(p=0.4),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # default dev and std for pretrained models
    #     ]),
    #     test_transform=transforms.Compose([
    #         transforms.Resize(256), 
    #         transforms.CenterCrop(224), # crop centrale
    #         transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # default dev and std for pretrained models
    #     ])
    # )

    # check corretto caricamento dataset
    # dataset_v1.show_info()

    # faccio un SCRIVI PROCEDURA TECNICA COME SI CHIAMA CHE HAI LETTO NEL LIBRO PER CERCARE IL MIGLIOR LEARNING RATE PER 10 EPOCHE ANCHE SE SONO POCHE, VEDIAMO LA DIFFERENZA TRA I MODELLI
    # DATI I COSTI COMPUTAZIONALI MOLTO LUNGHI
    # num workers = 2 li tengo bassi così non rompo i processi che girano nel pc in generale.
    # FAI IL CONTO PER IL NUMERO DI BATCH DELLO SCREEN DELL'IPHONE


    ######### PRIMO TRAINING #########
    # train(creator=CCMobileNetV2(), model_name=get_model_name(model_name="MobileNetV2", lr="0.001"), dataset=dataset_v1, output_class=3, batch_size=64, num_workers=2, drop_last=False, lr=0.001, epochs=10, save_each_iter=2)
    # train(creator=CCMobileNetV2(), model_name=get_model_name(model_name="MobileNetV2", lr="0.01"), dataset=dataset_v1, output_class=3, batch_size=64, num_workers=2, drop_last=False, lr=0.01, epochs=10, save_each_iter=2)
    # train(creator=CCAlexNet(), model_name=get_model_name(model_name="AlexNet", lr="0.001"), dataset=dataset_v1, output_class=3, batch_size=64, num_workers=2, drop_last=False, lr=0.001, epochs=10, save_each_iter=2)
    # train(creator=CCAlexNet(), model_name=get_model_name(model_name="AlexNet", lr="0.01"), dataset=dataset_v1, output_class=3, batch_size=64, num_workers=2, drop_last=False, lr=0.01, epochs=10, save_each_iter=2)
    # train(creator=CCVgg16(), model_name=get_model_name(model_name="Vgg16", lr="0.001"), dataset=dataset_v1, output_class=3, batch_size=64, num_workers=2, drop_last=False, lr=0.001, epochs=10, save_each_iter=2)
    # train(creator=CCVgg16(), model_name=get_model_name(model_name="Vgg16", lr="0.01"), dataset=dataset_v1, output_class=3, batch_size=64, num_workers=2, drop_last=False, lr=0.01, epochs=10, save_each_iter=2)
    # print("*** addestramento finito !!!! *****")
    ######### END PRIMO TRAINING #########



    ###### secondo training ########

    # applico qualche modifica al dataset
    # dataset_v2 = import_dataset('dataset', 
    #     train_transform=transforms.Compose([
    #         transforms.Resize(256),
    #         # inserire autoAugment ???
    #         transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN),
    #         transforms.RandomCrop(224),
    #         transforms.RandomHorizontalFlip(p=0.2),
    #         transforms.RandomPerspective(p=0.4),
    #         transforms.RandomVerticalFlip(p=0.3),
    #         transforms.RandomApply(torch.nn.ModuleList([
    #             transforms.Grayscale(num_output_channels=3), # tutti i modelli richiedono un'immagine a tre livelli
    #         ]), p=0.3), # effettuo un grayscale con probabilità 0.3
            
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # default dev and std for pretrained models
    #     ]),
    #     test_transform=transforms.Compose([
    #         transforms.Resize(256), 
    #         transforms.CenterCrop(224), # crop centrale
    #         transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomApply(torch.nn.ModuleList([
    #             transforms.Grayscale(num_output_channels=3), # tutti i modelli richiedono un'immagine a tre livelli
    #         ]), p=0.1), # effettuo un grayscale con probabilità 0.1
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # default dev and std for pretrained models
    #     ])
    # )

    
    # riprendo il training per i modelli che hanno avuto un valore migliore,
    # abilito il drop-last a differenza di prima ma lascio lo stesso numero di batch

    # cambio il dataset per fare generalizzare meglio il modello
    ## train(creator=CCAlexNet(), model_name=get_model_name(model_name="AlexNet_2dst", lr="0.001"), dataset=dataset_v2, output_class=3, batch_size=64, num_workers=2, drop_last=True, lr=0.001, epochs=10, save_each_iter=2,
    ##         loaded_model=join('models/AlexNet__lr=0.001', 'AlexNet__lr=0.001-10.pth'),
    ##         train_from_epoch=10, resume_global_step_from=46575 ) # indicatore che e' stato precedentemente trainato

    ## train(creator=CCMobileNetV2(), model_name=get_model_name(model_name="MobileNetV2_2dst", lr="0.001"), dataset=dataset_v2, output_class=3, batch_size=64, num_workers=2, drop_last=True, lr=0.001, epochs=10, save_each_iter=2,
    ##     loaded_model=join('models/MobileNetV2__lr=0.001', 'MobileNetV2__lr=0.001-10.pth'), train_from_epoch=10, resume_global_step_from=46575) # indicatore che e' stato precedentemente trainato

    ## # utilizzo lo dataset_v1 come fatto per le precedenti epoche
    ## train(creator=CCAlexNet(), model_name=get_model_name(model_name="AlexNet", lr="0.001"), dataset=dataset_v1, output_class=3, batch_size=64, num_workers=2, drop_last=True, lr=0.001, epochs=10, save_each_iter=2,
    ##         loaded_model=join('models/AlexNet__lr=0.001', 'AlexNet__lr=0.001-10.pth'), train_from_epoch=10, resume_global_step_from=46575) # indicatore che e' stato precedentemente trainato

    ## train(creator=CCMobileNetV2(), model_name=get_model_name(model_name="MobileNetV2", lr="0.001"), dataset=dataset_v1, output_class=3, batch_size=64, num_workers=2, drop_last=True, lr=0.001, epochs=10, save_each_iter=2,
    ##     loaded_model=join('models/MobileNetV2__lr=0.001', 'MobileNetV2__lr=0.001-10.pth'), train_from_epoch=10, resume_global_step_from=46575) # indicatore che e' stato precedentemente trainato


    ######## terzo training #########


    # decremento un po' il LR
    # salvo su un log diverso di tensorboard, quando serve guardare il graifico completo sposto i file all'interno e vedo tutti i dati

    ### alexNet è il metodo migliore! provo a vedere cosa cambia se abbasso un po' il LR sia con la versione
    ## trained su due dataset sia su quello su un terzo dataset leggermente modificato

    # applico qualche modifica al dataset
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

    # partendo dallo stesso numero di iterazioni controllo se abbassare il LR va bene in entrambi i casi di alexNet testati (con 1 dst solo e con 2 dataset)
    # per 20 epoche partendo dalla 20esima iterazione per entrambI!

    # LR = 0.001
    # dataset v1 
    train(creator=CCAlexNet(), model_name=get_model_name(model_name="AlexNet", lr="0.001"), dataset=dataset_v3, output_class=3, batch_size=64, num_workers=2, drop_last=True, lr=0.001, epochs=20, save_each_iter=4,
            loaded_model=join('models/AlexNet__lr=0.001', 'AlexNet__lr=0.001-20.pth'),
            train_from_epoch=20, resume_global_step_from=92655) # indicatore che e' stato precedentemente trainato
    
    # dataset v1 + v2
    train(creator=CCAlexNet(), model_name=get_model_name(model_name="AlexNet_2dst", lr="0.001"), dataset=dataset_v3, output_class=3, batch_size=64, num_workers=2, drop_last=True, lr=0.001, epochs=20, save_each_iter=4,
        loaded_model=join('models/AlexNet_2dst__lr=0.001', 'AlexNet_2dst__lr=0.001-20.pth'),
        train_from_epoch=20, resume_global_step_from=92655) # indicatore che e' stato precedentemente trainato


    # LR = 0.0003
    # dataset v1 
    train(creator=CCAlexNet(), model_name=get_model_name(model_name="AlexNet", lr="0.0003"), dataset=dataset_v3, output_class=3, batch_size=64, num_workers=2, drop_last=True, lr=0.0003, epochs=20, save_each_iter=4,
            loaded_model=join('models/AlexNet__lr=0.001', 'AlexNet__lr=0.001-20.pth'),
            train_from_epoch=20, resume_global_step_from=92655) # indicatore che e' stato precedentemente trainato
    
    # dataset v1 + v2
    train(creator=CCAlexNet(), model_name=get_model_name(model_name="AlexNet_2dst", lr="0.0003"), dataset=dataset_v3, output_class=3, batch_size=64, num_workers=2, drop_last=True, lr=0.0003, epochs=20, save_each_iter=4,
        loaded_model=join('models/AlexNet_2dst__lr=0.001', 'AlexNet_2dst__lr=0.001-20.pth'),
        train_from_epoch=20, resume_global_step_from=92655) # indicatore che e' stato precedentemente trainato

    print("\n\n---- Seconda procedura di test completata ------\n\n")

    # bisogna fare una differenza tra questi 4 per vedere quale performa meglio
    # per relazione: fai confronto tra tutti questi grafici per spiegare a cosa ti ha portato continuare a fare il training per piu epoche con il modello che hai edtto tu