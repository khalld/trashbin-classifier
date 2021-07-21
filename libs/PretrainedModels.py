
from __future__ import annotations
from abc import ABC, abstractmethod

from torch.utils.data.dataloader import DataLoader # https://docs.python.org/3/library/abc.html
from libs.TDContainer import TDContainer 
import time
from sklearn.metrics import accuracy_score  # computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
from os.path import join
import numpy as np

import torch
from torch import nn    # basic building-blocks for graphs https://pytorch.org/docs/stable/nn.html
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

# *** torchvision pretrained models https://pytorch.org/vision/stable/models.html ***
from torchvision.models import squeezenet1_0
from torchvision.models import alexnet
from torchvision.models import vgg16

class AverageValueMeter():
    """Calculate Average Value Meter"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.num = 0

    def add(self, value, num):
        self.sum += value*num
        self.num += num

    def value(self):
        try:
            return self.sum/self.num
        except:
            return None

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

    def trainval_classifier(self, model_name='experiment', lr: float=0.01, epochs: int=10, momentum: float=0.99, log_dir: str='logs', model_dir: str='models',
                            train_from_epoch: int=0, save_on_runtime: bool=False, save_each_iter: int=20):
        
        model = self.model
        timer_start = time.time()    
        
        criterion = nn.CrossEntropyLoss() # used for classification https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        
        optimizer = SGD(model.parameters(), lr, momentum=momentum)

        # meters
        loss_meter = AverageValueMeter()
        acc_meter = AverageValueMeter()

        # writer
        writer = SummaryWriter(join(log_dir, model_name))

        model.to(self.device)
        ## definiamo un dizionario contenente i loader di training e test
        loader = {
            'train': self.dst.training_loader,
            'validation': self.dst.validation_loader
        }
        global_step = 0
        for e in range(epochs):
            print ("\rComputing: %d/%d" % ( e+1,  epochs), end="") # \r allow to make carriage returns

            for mode in ['train', 'validation']:
                loss_meter.reset(); acc_meter.reset()
                model.train() if mode == 'train' else model.eval()
                with torch.set_grad_enabled(mode=='train'): # abilitiamo i gradienti o solo in training
                    for i, batch in enumerate(loader[mode]):
                        x = batch[0].to(self.device) # portiamoli su device corretto
                        y = batch[1].to(self.device)
                        output = model(x)

                        # aggiorniamo il global_step
                        # conterrà il numero di campioni visti durante il training
                        n = x.shape[0]  # n di elementi nel batch
                        global_step += n
                        l = criterion(output, y)

                        if mode == 'train':
                            l.backward()
                            optimizer.step()
                            optimizer.zero_grad()

                        acc = accuracy_score(y.to('cpu'), output.to('cpu').max(1)[1])
                        loss_meter.add(l.item(), n)
                        acc_meter.add(acc,n)

                        # loggiamo i risultati iterazione per iterazione solo durante il training
                        if mode == 'train':
                            writer.add_scalar('loss/train', loss_meter.value(), global_step=global_step)
                            writer.add_scalar('accuracy/train', acc_meter.value(), global_step=global_step)

                    # una volta finita l'epoca sia nel caso di training che di test loggiamo le stime finali
                    writer.add_scalar('loss/' + mode, loss_meter.value(), global_step=global_step)
                    writer.add_scalar('accuracy/' + mode, acc_meter.value(), global_step=global_step)

            # conserviamo i pesi del modello alla fine di un ciclo di training e test..
            # ...sul runtime
            if save_on_runtime is True:
                torch.save(model.state_dict(), '%s-%d.pth'%(model_name, (e+1) + train_from_epoch ) )

            # ...ogni save_each_iter salvo il modello sul drive per evitare problemi di spazio su Gdrive
            if ((e+1) % save_each_iter == 0 or (e+1) % 50 == 0):
                torch.save(model.state_dict(), model_dir + '%s-%d.pth'%(model_name, (e+1) + train_from_epoch ) )

        timer_end = time.time()
        print("Ended in: ", ((timer_end - timer_start) / 60 ), "minutes" )
        return model

    def test_classifier(self, model, dataLoader: DataLoader) -> None:  # self.dataLoader
        model.to(self.device)
        predictions, labels = [], []
        for batch in dataLoader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            output = model(x)
            preds = output.to('cpu').max(1)[1].numpy()
            labs = y.to('cpu').numpy()
            predictions.extend(list(preds))
            labels.extend(list(labs))
        return np.array(predictions), np.array(labels)

    def train(self, model_name: str, lr: float=0.01, epochs: int=10, momentum: float=0.99, log_dir: str='logs', model_dir: str='models', train_from_epoch: int=0, save_on_runtime: bool=False, save_each_iter: int=20) -> None:

        self.model_finetuned = self.trainval_classifier(model_name=model_name, lr=lr, epochs=epochs, momentum=momentum,
                                                        log_dir=log_dir, model_dir=model_dir,
                                                        train_from_epoch=train_from_epoch, save_on_runtime=save_on_runtime, save_each_iter=save_each_iter )
        print("**** Training procedure ended. Start to calculate accuracy ... ****")

        self.model_finetuned_predictions_test, self.dataset_labels_test = self.test_classifier(self.model_finetuned, self.dst.test_loader)
        print("Accuracy of %s : %0.2f%%" % (model_name, (accuracy_score(self.dataset_labels_test, self.model_finetuned_predictions_test)*100)) )

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
        model = vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, output_class)

        return model