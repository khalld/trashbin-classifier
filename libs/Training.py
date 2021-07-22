import time
from sklearn.metrics import accuracy_score  # computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
from os.path import join
import numpy as np

import torch
from torch import nn    # basic building-blocks for graphs https://pytorch.org/docs/stable/nn.html
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from libs.TDContainer import TDContainer

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

def trainval_test(loader: TDContainer):
    print(loader)
    print(loader.training_loader, type(loader.training_loader))
    print(loader.test_loader, type(loader.test_loader))
    print(loader.validation_loader, type(loader.validation_loader))

def trainval_classifier(model, dst_container: TDContainer, model_name='experiment', lr: float=0.01, epochs: int=10, momentum: float=0.99, logdir: str='logs', model_dir: str='models', train_from_epoch: int=0, save_on_runtime: bool=False, save_each_iter: int=20):
    timer_start = time.time()    
    criterion = nn.CrossEntropyLoss() # used for classification https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    optimizer = SGD(model.parameters(), lr, momentum=momentum)

    # meters
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()

    # writer
    writer = SummaryWriter(join(logdir, model_name))

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    ## definiamo un dizionario contenente i loader di training e test
    loader = {
        'train': dst_container.training_loader,
        'validation': dst_container.validation_loader
    }
    global_step = 0
    for e in range(epochs):
        print ("\rComputed: %d/%d, current: loss: %s accuracy: %s" % ( e+1,  epochs, loss_meter.value(), acc_meter.value()), end="") # \r allow to make carriage returns
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
            torch.save(model.state_dict(), '%s-%d.pth'%(model_name, (e+1) + train_from_epoch ) )

        # ...ogni save_each_iter salvo il modello sul drive per evitare problemi di spazio su Gdrive
        if ((e+1) % save_each_iter == 0 or (e+1) % 50 == 0):
            torch.save(model.state_dict(), model_dir + '%s-%d.pth'%(model_name, (e+1) + train_from_epoch ) )

    timer_end = time.time()
    print("\nEnded in: ", ((timer_end - timer_start) / 60 ), "minutes" )
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