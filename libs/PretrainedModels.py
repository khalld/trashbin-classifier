
from __future__ import annotations
from __future__ import print_function 
from __future__ import division
from abc import ABC, abstractmethod

from libs.TDContainer import TDContainer  ## run local
# # from TDContainer import TDContainer ## run colab

# import torch
# from torch import nn    # basic building-blocks for graphs https://pytorch.org/docs/stable/nn.html

# # *** torchvision pretrained models https://pytorch.org/vision/stable/models.html ***
# from torchvision.models import alexnet
# from torchvision.models import vgg16
# from torchvision.models import mobilenet_v2

# DA UNCOMMENTARE SOPRA ^^^^

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
        self.model, self.input_size = product.get_model(num_classes, feature_extract, use_pretrained)

    # self.device è deprecato!!!
    # def load_model(self, path: str) -> None:
    #     print("Loading model using load_state_dict..")
    #     if (self.device == "cpu"):
    #         self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False) # su colab c'è la cpu qui no! quindi se lo alleno sulla gpu devo cambiarlo
    #     else:
    #         self.model.load_state_dict(torch.load(path), strict=False) # VEDI COSA SIGNIFICA STRICT

    def get_info(self) -> None:
        print("Model:\n", self.model)
        print("Input size:\n", self.input_size)

    # non so se sono necessari
    def get_model(self):
        return self.model

    def get_input_size(self):
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

        return model_ft, input_size

class SqueezeNet_cp(PretrainedModel):
    def get_model(self, num_classes: int, feature_extract: bool = True, use_pretrained: bool=True):
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

        return model_ft, input_size

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

        return model_ft, input_size

