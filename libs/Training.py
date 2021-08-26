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

# 
class AvgMeter():
    """ Calculates the loss and accuracy on individual batches"""
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

def train(model, dst_container: TDContainer, criterion, optimizer,
                        epochs: int=10, 
                        train_from_epoch: int=0, save_each_iter: int=20, model_name: str='experiment',
                        resume_global_step_from: int=0):
    
    logdir = 'logs'
    modeldir = 'models'

    time_start = time.time()

    # meters
    loss_meter = AvgMeter()
    acc_meter = AvgMeter()

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

        if ((e+1) % save_each_iter == 0 ):
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