
from __future__ import annotations
from abc import ABC, abstractmethod

from libs.TDContainer import TDContainer  ## run local
# from TDContainer import TDContainer ## run colab

import torch
from torch import nn    # basic building-blocks for graphs https://pytorch.org/docs/stable/nn.html

# *** torchvision pretrained models https://pytorch.org/vision/stable/models.html ***
from torchvision.models import alexnet
from torchvision.models import vgg16
from torchvision.models import mobilenet_v2

# Creator
class PretrainedModelsCreator(ABC):
    """The Creator class declares the factory method that is supposed to return an
    object of a Product class. The Creator's subclasses usually provide the
    implementation of this method."""

    @abstractmethod
    def factory_method(self):
        """No default implementation needed"""
        pass

    def initialize_model(self, output_class: int = 2):
        """
            Nasce dalla necessità che a livello di GUI non devo inizializzare nessun dataset né dataLoader ma semplicemente devo scaricare il modello
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        product = self.factory_method()
        self.model = product.get_model(output_class)

        return True

    def initialize_dst(self, dataset: TDContainer, output_class: int = 2, batch_size: int=32, num_workers: int=2, drop_last: bool=False) -> None:
        """The Creator's primary responsibility is not creating products. Usually, it contains 
        some core business logic that relies on Product objects, returned by the factory method.
        Subclasses can indirectly change that business logic by overriding the
        factory method and returning a different type of product from it."""

        # implementa nel modulo
        # use cpu if is possible
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        # call factory method to create a Product object
        product = self.factory_method()
        # get the model from product
        self.model = product.get_model(output_class)
        # set the dataset inside the object
        self.dst = dataset
        ## instantiate DataLoader too
        self.dst.create_data_loader(batch_size=batch_size, num_workers=num_workers, drop_last=drop_last )

    def load_model(self, path: str) -> None:
        print("Loading model using load_state_dict..")
        if (self.device == "cpu"):
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))) # su colab c'è la cpu qui no! quindi se lo alleno sulla gpu devo cambiarlo
        else:
            self.model.load_state_dict(torch.load(path)) 

    def get_info(self) -> None:
        print("Information about model", self.model)

    def get_state_dict(self) -> None:
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

    def get_parameters(self) -> None:
        print("Print parameter")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print("name", name, "param", param.data)

    def return_model(self):
        return self.model

""" Concrete Creators override the factory method in order to change the resulting product's type. """

class CCAlexNet(PretrainedModelsCreator):
    def factory_method(self) -> PretrainedModel:
        return CPAlexNet()

class CCAlexNet_v2(PretrainedModelsCreator):
    def factory_method(self):
        return CPAlexNet_v2()

class CCVgg16(PretrainedModelsCreator):
    def factory_method(self) -> PretrainedModel:
        return CPVgg16()

class CCMobileNetV2(PretrainedModelsCreator):
    def factory_method(self) -> PretrainedModel:
        return CPMobileNetV2()

"""Product"""
class PretrainedModel(ABC):
    """ The Product interface declares the operations that all concrete products
    must implement."""
    @abstractmethod
    def get_model(self, output_class: int = 3):
        pass

"""Concrete Products provide various implementations of the Product interface."""
class CPAlexNet(PretrainedModel):
    def get_model(self, output_class: int = 3):
        model = alexnet(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        model.classifier[6] = nn.Linear(4096)

class CPAlexNet(PretrainedModel):
    def get_model(self, output_class: int = 3):
        model = alexnet(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        model.classifier[6] = nn.Sequential(
                        nn.Linear(4096, 256),
                        nn.SiLU(),  # better than reLu
                        ## dropout e applicato solitamente dopo la funzione di attivazione non lineare, in alcuni casi con la relu ha piu senso il contrario
                        nn.Dropout(0.4),    # effective technique for regularization and preventing the co-adaptation of neurons
                        nn.Linear(256, output_class)
                    )


        return model

class CPVgg16(PretrainedModel):
    def get_model(self, output_class: int = 3):

        # load pretrained weights from a network trained on large dataset 
        model = vgg16(pretrained=True)

        # initially freeze all the models weights
        for param in model.parameters():
            param.requires_grad = False

        # add custom classifier
        # model.classifier[6] = nn.Linear(4096, output_class)
        # by default requires_grad = True
        model.classifier[6] = nn.Sequential(
                                nn.Linear(4096, 256),
                                nn.SiLU(),  # better than reLu
                                nn.Dropout(0.4),    # effective technique for regularization and preventing the co-adaptation of neurons
                                nn.Linear(256, output_class)
                            )

        return model

class CPMobileNetV2(PretrainedModel):
    def get_model(self, output_class: int = 3):
        model = mobilenet_v2(pretrained=True)

        # initialy freeze all the models weights
        for param in model.parameters():
            param.requires_grad = False

        # add custom classifier
        model.classifier = nn.Sequential(
                                nn.Dropout(p=0.2, inplace=False),   # resta uguale
                                nn.Linear(in_features=1280, out_features=output_class, bias=True)
                            )

        return model