from __future__ import annotations
from abc import ABC, abstractmethod # https://docs.python.org/3/library/abc.html

from torchvision.models import squeezenet1_0
from torchvision.models import alexnet
from torchvision.models import vgg16
from torch import nn

# Creator
class PretrainedModelsCreator(ABC):
    """The Creator class declares the factory method that is supposed to return an
    object of a Product class. The Creator's subclasses usually provide the
    implementation of this method."""

    @abstractmethod
    def factory_method(self):
        """ No default implementation needed"""
        pass

    def initialize_dst(self, dataset, output_class: int = 3) -> None:
        """
        The Creator's primary responsibility is not creating products. Usually, it contains
        some core business logic that relies on Product objects, returned by the factory method.
        Subclasses can indirectly change that business logic by overriding the
        factory method and returning a different type of product from it.
        """

        # call factory method to create a Product object
        product = self.factory_method()
        # get the model from product
        self.model = product.get_model(output_class)
        # set the dataset inside the object
        self.dataset = dataset

    def train(self) -> None:
        print("Training procedure TODO...")

    def load_model(self, path: str) -> None:
        # self.model.load_state_dict(torch.load(path))
        print("Loading model from " + path + "TODO...")

    def get_info(self) -> None:
        print("Information about model:\n", self.model)

"""
    Concrete Creators override the factory method in order to change the resulting product's type.
"""

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

# Product
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
        model = vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, output_class)

        return model


def client_code(creator: PretrainedModelsCreator, dataset) -> None:

    creator.initialize_dst(dataset, 9)
    creator.get_info()
    print("\n\n ---", creator.train())