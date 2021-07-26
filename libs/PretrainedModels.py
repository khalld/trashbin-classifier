
from __future__ import annotations
from abc import ABC, abstractmethod

from TDContainer import TDContainer 

import torch
from torch import nn    # basic building-blocks for graphs https://pytorch.org/docs/stable/nn.html

# *** torchvision pretrained models https://pytorch.org/vision/stable/models.html ***
from torchvision.models import squeezenet1_0
from torchvision.models import alexnet
from torchvision.models import vgg16


# Creator
class PretrainedModelsCreator(ABC):
    """The Creator class declares the factory method that is supposed to return an
    object of a Product class. The Creator's subclasses usually provide the
    implementation of this method."""

    @abstractmethod
    def factory_method(self):
        """No default implementation needed"""
        pass

    def initialize_dst(self, dataset: TDContainer, output_class: int = 2, batch_size: int=32, num_workers: int=2, drop_last: bool=False) -> None:
        """The Creator's primary responsibility is not creating products. Usually, it contains 
        some core business logic that relies on Product objects, returned by the factory method.
        Subclasses can indirectly change that business logic by overriding the
        factory method and returning a different type of product from it."""

        # implementa nel modulo
        # use cpu if is possible
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
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
        self.model.load_state_dict(torch.load(path))

    def get_info(self) -> None:
        print("Information about model:\n", self.model)

    def get_parameter(self) -> None:
        print("Print parameter function todo....")

""" Concrete Creators override the factory method in order to change the resulting product's type. """
# ConcreteCreator1 
class CCSqueezeNet(PretrainedModelsCreator):
    def factory_method(self) -> PretrainedModel:
        return CPSqueezeNet()

# ConcreteCreator2
class CCAlexNet(PretrainedModelsCreator):
    def factory_method(self) -> PretrainedModel:
        return CPAlexNet()

# ConcreteCreator3
class CCVgg16(PretrainedModelsCreator):
    def factory_method(self) -> PretrainedModel:
        return CPVgg16()

"""Product"""
class PretrainedModel(ABC):
    """ The Product interface declares the operations that all concrete products
    must implement."""
    @abstractmethod
    def get_model(self, output_class: int = 3):
        pass

"""Concrete Products provide various implementations of the Product interface."""
# ConcreteProduct1
class CPSqueezeNet(PretrainedModel):
    def get_model(self, output_class: int = 3):
        model = squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, output_class, kernel_size=(1,1), stride=(1,1))

        return model
    
# ConcreteProduct2
class CPAlexNet(PretrainedModel):
    def get_model(self, output_class: int = 3):
        model = alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(4096, output_class) # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

        return model

# ConcreteProduct3
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