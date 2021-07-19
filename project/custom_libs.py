import numpy as np
import pandas as pd
from PIL import Image
from os.path import join, splitext
import time
from sklearn.metrics import accuracy_score  # computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

import torch
from torch.utils import data # necessary to create a map-style dataset https://pytorch.org/docs/stable/data.html
from torch import nn    # basic building-blocks for graphs https://pytorch.org/docs/stable/nn.html
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

# *** torchvision pretrained models https://pytorch.org/vision/stable/models.html ***
from torchvision.models import squeezenet1_0
from torchvision.models import alexnet
from torchvision.models import vgg16

class TrashbeanDataset(data.Dataset): # data.Dataset https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset
    """ A map-style dataset class used to manipulate a dataset composed by:
        image path of trashbean and associated label that describe the available capacity of the trashbean
            0 : empty trashbean
            1 : half trashbean
            2 : full trashbean

        Attributes
        ----------
        data : str
            path of csv file
        transform : torchvision.transforms

        Methods
        -------
        __len__()
            Return the length of the dataset

        __getitem__(i)
            Return image, label of i element of dataset  
    """

    def __init__(self, csv=None, transform=None):
        """ Constructor of the dataset
            Parameters
            ----------
            csv : str
            path of the dataset

            transform : torchvision.transforms
            apply transform to the dataset

            Raises
            ------
            NotImplementedError
                If no path is passed is not provided a default dataset
        """
        
        if csv is None:
            raise NotImplementedError("No default dataset is provided")
        if splitext(csv)[1] != '.csv':
            raise NotImplementedError("Only .csv files are supported")
        
        self.data = pd.read_csv(csv)        # import from csv using pandas
        self.data = self.data.iloc[np.random.permutation(len(self.data))]       # random auto-permutation of the data
        self.transform = transform

    def __len__(self):
        """ Return length of dataset """
        return len(self.data)

    def __getitem__(self, i=None):
        """ Return the i-th item of dataset

            Parameters
            ----------
            i : int
            i-th item of dataset

            Raises
            ------
            NotImplementedError
            If i is not a int
        """
        if i is None:
            raise NotImplementedError("Only int type is supported for get the item. None is not allowed")
        
        im_path, im_label = self.data.iloc[i]['path'], self.data.iloc[i].label
        im = Image.open(im_path)        # Handle image with Image module from Pillow https://pillow.readthedocs.io/en/stable/reference/Image.html
        if self.transform is not None:
            im = self.transform(im)
        return im, im_label


class TDContainer:
    """ Class that contains the dataset for training, validation and test
        Attributes
        ----------
        self.training, self.validation, self.test are the TrashbeanDataset object
        self.training_loader, self.validation_loader, self.test_loader are DataLoader of the correspective TrashbeanDataset
    """

    def __init__(self, training=None, validation=None, test=None):
        """ Constructor of the class. Instantiate an Trashbean dataset for each dataset

            Parameters
            ----------
            training: str, required
                path of training dataset csv
            
            validation: str, required
                path of validation dataset csv
            
            test: str, required
                path of test dataset csv
        """
    
        if training is None or validation is None or test is None:
            raise NotImplementedError("No default dataset is provided")
        
        if isinstance(training, dict) is False or isinstance(validation, dict) is False or isinstance(test, dict) is False:
            raise NotImplementedError("Constructor accept only dict file.")

        if training['path'] is None or validation['path'] is None or test['path'] is None or isinstance(training['path'], str) is False or isinstance(validation['path'], str) is False or isinstance(test['path'], str) is False:
            raise NotImplementedError("Path file is required and need to be a str type.")

        self.training = TrashbeanDataset(training['path'], transform=training['transform'])
        self.validation = TrashbeanDataset(validation['path'], transform=validation['transform'])
        self.test = TrashbeanDataset(test['path'], transform=test['transform'])
        self.hasDl = False

    def create_data_loader(self, _batch_size=32, _num_workers=2):
        """ Create data loader for each dataset

            https://pytorch.org/docs/stable/data.html
            
            Parameters
            ----------

            _batch_size: int
                number of batches, default 32

            _num_workers: int
                number of workers
        """

        if isinstance(_batch_size, int) is False or isinstance(_num_workers, int) is False:
            raise NotImplementedError("Parameters accept only int value.")

        self.training_loader = DataLoader(self.training, batch_size=_batch_size, num_workers=_num_workers, shuffle=True)
        self.validation_loader = DataLoader(self.validation, batch_size=_batch_size, num_workers=_num_workers)
        self.test_loader = DataLoader(self.test, batch_size=_batch_size, num_workers=_num_workers)
        self.hasDl = True

    def show_info(self):
        """ Print info of dataset """
        print("\n=== *** DB INFO *** ===")
        print("Training:", self.training.__len__(), "values, \nValidation:", self.validation.__len__(), "values, \nTest:", self.test.__len__())
        print("DataLoader:", self.hasDl)
        print("=== *** END *** ====\n")

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

def trainval_classifier(model, train_loader, validation_loader, exp_name='experiment',
                        lr=0.01, epochs=10, momentum=0.99, train_from_epoch=0,
                        log_dir='/content/gdrive/MyDrive/trashbean-classifier/logs', models_dir='/content/gdrive/MyDrive/trashbean-classifier/logs/models/',
                        save_on_runtime=False):
    timer_start = time.time()    
    
    criterion = nn.CrossEntropyLoss() # used for classification https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    
    optimizer = SGD(model.parameters(), lr, momentum=momentum)

    # meters
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()

    # writer
    writer = SummaryWriter(join(log_dir, exp_name))

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    ## definiamo un dizionario contenente i loader di training e test
    loader = {
        'train': train_loader,
        'validation': validation_loader
    }
    global_step = 0
    print("Computing epoch:")
    for e in range(epochs):
        print(e+1, "/", epochs, "... ")
        # iteriamo tra due modalità: train e test
        for mode in ['train', 'validation']:
            loss_meter.reset(); acc_meter.reset()
            model.train() if mode == 'train' else model.eval()
            with torch.set_grad_enabled(mode=='train'): # abilitiamo i gradienti o solo in training
                for i, batch in enumerate(loader[mode]):
                    x = batch[0].to(device) # portiamoli su device corretto
                    y = batch[1].to(device)
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
            torch.save(model.state_dict(), '%s-%d.pth'%(exp_name, (e+1) + train_from_epoch ) )

        # ...ogni 20 epoche salvo il modello sul drive per evitare problemi di spazio su Gdrive
        if ((e+1) % 20 == 0 or (e+1) % 50 == 0):
            #torch.save(model.state_dict(), '/content/gdrive/MyDrive/trashbean-classifier/logs/models/%s-%d.pth'%(exp_name, (e+1) + train_from_epoch ) )
            torch.save(model.state_dict(), models_dir + '%s-%d.pth'%(exp_name, (e+1) + train_from_epoch ) )

    timer_end = time.time()
    print("Ended in: ", ((timer_end - timer_start) / 60 ), "minutes" )
    return model

def test_classifier(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    predictions, labels = [], []
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        output = model(x)
        preds = output.to('cpu').max(1)[1].numpy()
        labs = y.to('cpu').numpy()
        predictions.extend(list(preds))
        labels.extend(list(labs))
    return np.array(predictions), np.array(labels)


### è qui che devo implementare il design pattern!!!
class Pretrained_models:
    def __init__(self, dataset, num_class):
        self.dataset = dataset
        self.num_class = num_class

    def get_squeezenet(self):
        self.model = squeezenet1_0(pretrained=True)
        self.model.classifier[1] = nn.Conv2d(512, self.num_class, kernel_size=(1,1), stride=(1,1))

    def get_alexnet(self):
        self.model = alexnet(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, self.num_class) # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

    def get_vgg16(self):  # no batch mode
        self.model = vgg16(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, self.num_class)

    def finetuning_and_accuracy(self, _exp_name, _lr, _epochs, _train_from_epoch ):

        if self.dataset.hasDl is False:
            raise NotImplementedError("Need to instantiate the dataLoader before. Use TDContainer.create_data_loader(batches, workers)")
        
        print("*** Training procedure started, please wait ....")
        self.model_finetuned = trainval_classifier(self.model, self.dataset.training_loader, self.dataset.validation_loader, exp_name=_exp_name, lr=_lr, epochs=_epochs, train_from_epoch=_train_from_epoch)
        print("**** Training procedure ended. Start to calculate accuracy ...")
        self.model_finetuned_predictions_test, self.dataset_labels_test = test_classifier(self.model_finetuned, self.dataset.test_loader)
        print("Accuracy of " + _exp_name + " %0.2f%%" % (accuracy_score(self.dataset_labels_test, self.model_finetuned_predictions_test)*100) )

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def print_parameters(self):
        print("Find parameters: ", self.model.parameters())

        for param in self.model.parameters():
            print(param)

    def info(self):
        print(self.model)