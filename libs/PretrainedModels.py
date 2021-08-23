
from __future__ import annotations
from __future__ import print_function 
from __future__ import division
from abc import ABC, abstractmethod

from libs.TDContainer import TDContainer  
from libs.Training2 import train_model_adapted
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# helper function
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Creator
class PretrainedModelsCreator(ABC):
    """The Creator class declares the factory method that is supposed to return an
    object of a Product class. The Creator's subclasses usually provide the
    implementation of this method."""

    @abstractmethod
    def factory_method(self):
        """No default implementation needed"""
        pass

    # cosa da fare all'inizio
    def init_model(self, num_classes: int = 3, feature_extract: bool=True, use_pretrained: bool = True):
        """
            Nasce dalla necessità che a livello di GUI non devo inizializzare nessun dataset né dataLoader ma semplicemente devo scaricare il modello
        """
        product = self.factory_method()
        self.feature_extract = feature_extract
        self.model_ft, self.input_size, self.is_inception = product.get_model(num_classes, feature_extract, use_pretrained)

    def do_train(self, dataset, num_epochs, lr, momentum, criterion):
        print('Feature extract is setted to: ', self.feature_extract)

        # Create optimizer
        params_to_update = self.model_ft.parameters()
        
        if self.feature_extract:
            params_to_update = []
            for name,param in self.model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in self.model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

        optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum)
        # End create optimizer

        # criterion = nn.CrossEntropyLoss()
        model_tr, history = train_model_adapted(model=self.model_ft, dst_container=dataset, criterion=criterion, optimizer=optimizer, num_epochs=num_epochs, is_inception=self.is_inception )

        return model_tr, history

    # self.device è deprecato!!!
    # def load_model(self, path: str) -> None:
    #     print("Loading model using load_state_dict..")
    #     if (self.device == "cpu"):
    #         self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False) # su colab c'è la cpu qui no! quindi se lo alleno sulla gpu devo cambiarlo
    #     else:
    #         self.model.load_state_dict(torch.load(path), strict=False) # VEDI COSA SIGNIFICA STRICT

    def get_info(self) -> None:
        print("Finetuned model info:\n", self.model_ft)
        print("Input size:\n", self.input_size)

    # non so se sono necessari
    def ret_model(self):
        return self.model_ft

    def ret_input_size(self):
        return self.input_size

""" Concrete Creators override the factory method in order to change the resulting product's type. """
class AlexNet_cc(PretrainedModelsCreator):
    def factory_method(self) -> PretrainedModel:
        return AlexNet_cp()
        
class SqueezeNet_cc(PretrainedModelsCreator):
    def factory_method(self) -> PretrainedModel:
        return SqueezeNet_cp()

class InceptionV3_cc(PretrainedModelsCreator):
    def factory_method(self) -> PretrainedModel:
        return InceptionV3_cp()

"""Product"""
class PretrainedModel(ABC):
    """ The Product interface declares the operations that all concrete products
    must implement."""
    @abstractmethod
    def get_model(self, num_classes: int = 3, feature_extract: bool=True, use_pretrained: bool=True):
        pass

"""Concrete Products provide various implementations of the Product interface."""
class AlexNet_cp(PretrainedModel):
    def get_model(self, num_classes: int = 3, feature_extract: bool = True, use_pretrained: bool=True):

        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

        is_inception = False

        return model_ft, input_size, is_inception

class SqueezeNet_cp(PretrainedModel):
    def get_model(self, num_classes: int, feature_extract: bool = True, use_pretrained: bool=True):
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

        is_inception = False

        return model_ft, input_size, is_inception

class InceptionV3_cp(PretrainedModel):
    def get_model(self, num_classes: int, feature_extract: bool = True, use_pretrained: bool=True):
        """Be careful, expects (299,299) sized images and has auxiliary output """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

        is_inception = True

        return model_ft, input_size, is_inception

