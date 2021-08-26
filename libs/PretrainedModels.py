
from __future__ import annotations
from __future__ import print_function 
from __future__ import division
from abc import ABC, abstractmethod
from libs.TDContainer import TDContainer

from libs.Training import train
import torch.nn as nn
import torch.optim as optim
from torchvision import models

def set_parameter_requires_grad(model: models, feature_extracting: bool):
    """Helper function that sets the `require_grad` attribute of parameter in the model to False when is used feature extracting"""

    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class PretrainedModelsCreator(ABC):
    """The Creator class declares the factory method that return an object of a Product class."""

    @abstractmethod
    def factory_method(self):
        """No default implementation needed"""

        pass

    def init_model(self, model_name: str, num_classes: int = 3, feature_extract: bool=True, use_pretrained: bool = True):
        """Initialize the model"""

        product = self.factory_method()
        self.feature_extract = feature_extract
        self.model_name = model_name
        self.model_ft, self.input_size, self.is_inception = product.get_model(num_classes, feature_extract, use_pretrained)

    def do_train(self, dataset: TDContainer, num_epochs: int, lr: float, momentum: float, criterion: nn, train_from_epoch: int, save_each_iter: int, resume_global_step_from: int):
        """Make training of the current model"""

        if (self.feature_extract is True):
            print('Feature extracting')
        else:
            print('Fine tuning')
        
        # create an optimizer that allow to update
        params_to_update = self.model_ft.parameters()

        # If is choosed finetuning all parameters will be updated. 
        # if is choosed feature extract method, only parameters just initialized will be updated (with requires_grad=True)

        if self.feature_extract:
            params_to_update = []
            for name,param in self.model_ft.named_parameters():
                if param.requires_grad == True: 
                    params_to_update.append(param)
        #             print("\t",name)
        # else:
        #     for name,param in self.model_ft.named_parameters():
        #         if param.requires_grad == True:
        #             print("\t",name)

        optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum) # to optimize the parameter

        # model_tr, history = train_model(model=self.model_ft,
        #                                 dst_container=dataset,
        #                                 criterion=criterion,
        #                                 optimizer=optimizer,
        #                                 num_epochs=num_epochs,
        #                                 model_name=self.model_name,
        #                                 train_from_epoch=train_from_epoch,
        #                                 save_each_iter=save_each_iter,
        #                                 resume_global_step_from=resume_global_step_from,
        #                                 is_inception=self.is_inception)

        # return model_tr, history

        model_tr = train(model=self.model_ft, dst_container=dataset, criterion=criterion, 
                                        optimizer=optimizer, epochs=num_epochs, train_from_epoch=train_from_epoch, 
                                        save_each_iter=save_each_iter, model_name=self.model_name, resume_global_step_from=resume_global_step_from )

        return model_tr

    # self.device è deprecato!!!
    # def load_model(self, path: str) -> None:
    #     print("Loading model using load_state_dict..")
    #     if (self.device == "cpu"):
    #         self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False) # su colab c'è la cpu qui no! quindi se lo alleno sulla gpu devo cambiarlo
    #     else:
    #         self.model.load_state_dict(torch.load(path), strict=False) # VEDI COSA SIGNIFICA STRICT

    def get_info(self) -> None:
        """Get info of finetuned model"""

        print("Finetuned model info:\n", self.model_ft)
        print("Input size:\n", self.input_size)

    def ret_model(self):
        """Return finetuned model"""

        return self.model_ft

    def ret_input_size(self):
        """Return input size of the finetuned model"""

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

