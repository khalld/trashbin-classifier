from libs.TDContainer import TDContainer
import torch
import torch.optim as optim
import time
import copy
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import os
import torchvision
import numpy as np
from sklearn.metrics import accuracy_score # computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

def train_model(model: torchvision.models, dst_container: TDContainer, criterion: nn, optimizer: optim, num_epochs: int=25, model_name: str='experiment', train_from_epoch: int=0, save_each_iter: int=2, resume_global_step_from: int=0, is_inception: bool=False):
    """
        Parameters
        -----------
            model: required
                model to train
            dst_container: TDContainer, required
            criterion: nn, required
            optimizer: optim, required
            num_epochs: int, default 25
            model_name: str, default 'experiment'
            train_from_epoch: int, 
                allows you to save the model from a certain epoch useful when need to continue a training from .pth model
            save_each_iter: int,
                save .pth model each times
            resume_global_step_from: int
                keep writing on tensoboard from a specific point
            is_inception:
                used for training with inceptionv3
    """
    time_start = time.time()

    # meters
    val_acc_history = []

    # assignment statements in Python do not copy objects, they create bindings between a target and an object. Need to use deepcopy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    writer = SummaryWriter(os.path.join('logs', model_name))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # moving model to device
    model.to(device)

    # create dict with two different datasets
    dataloaders = {
        'train': dst_container.training_loader,
        'validation': dst_container.validation_loader
    }

    global_step = 0 + resume_global_step_from
    
    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('Epoch %d/%d' % (epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for mode in ['train', 'validation']:
            if mode == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_accuracy = 0

            # iterate over data
            for inputs, labels in dataloaders[mode]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # global_step contains number of element seen during the training
                global_step += inputs.shape[0]  # input.shape[0] number of training's elements

                # sets the gradients of all optimized tensors to zero
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(mode == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    # mode we calculate the loss by summing the final output and the auxiliary output
                    # but in testing we only consider the final output.
                    if is_inception and mode == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958

                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only in training phase
                    if mode == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_accuracy += torch.sum(preds == labels.data)

                # logs result for each iterations only during training
                if mode == 'train':
                    writer.add_scalar('loss/train', running_loss , global_step=global_step)
                    writer.add_scalar('accuracy/train', running_accuracy , global_step=global_step)


            epoch_loss = running_loss / len(dataloaders[mode].dataset)
            epoch_acc = running_accuracy.double() / len(dataloaders[mode].dataset)

            writer.add_scalar('loss/' + mode, epoch_loss, global_step=global_step)
            writer.add_scalar('accuracy/' + mode, epoch_acc , global_step=global_step)

            # log final values at the end of the epoch
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(mode, epoch_loss, epoch_acc))

            # deep copy the model with best weights in validation
            if mode == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if mode == 'validation':
                val_acc_history.append(epoch_acc)
        
        if ((epoch+1) % save_each_iter == 0):
            torch.save(model.state_dict(), os.path.join('models', '%s-%d.pth' %(model_name, (epoch+1)+train_from_epoch) )  )

    time_elapsed = time.time() - time_start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best value Accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

class AverageValueMeter():
    """ Calculate"""
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

def trainval_classifier(model, dst_container: TDContainer, criterion, optimizer,
                        epochs: int=10, 
                        train_from_epoch: int=0, save_each_iter: int=20, model_name: str='experiment',
                        resume_global_step_from: int=0):
    
    logdir = 'logs'
    modeldir = 'models'

    time_start = time.time()    
    # criterion = nn.CrossEntropyLoss() # used for classification https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    # optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

    # meters
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()

    # writer
    writer = SummaryWriter(os.path.join(logdir, model_name))

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    ## definiamo un dizionario contenente i loader di training e test
    loader = {
        'train': dst_container.training_loader,
        'validation': dst_container.validation_loader
    }
    global_step = 0 + resume_global_step_from
    for e in range(epochs):
        print('Epoch %d/%d' % (e, epochs - 1))
        print('-' * 10)

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
        
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(mode, loss_meter.value(), acc_meter.value()))

        # ...ogni save_each_iter salvo il modello sul drive per evitare problemi di spazio su Gdrive
        if ((e+1) % save_each_iter == 0 or (e+1) % 50 == 0):
            torch.save(model.state_dict(), modeldir + '/%s-%d.pth'%(model_name, (e+1) + train_from_epoch ) )

    time_elapsed = time.time() - time_start
    
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

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