import torch
import torch.optim as optim
import time
import copy
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import os

def train_model(model,
                        dst_container, 
                        criterion,
                        optimizer,
                        num_epochs=25, 
                        model_name='experiment',
                        train_from_epoch=0,
                        save_each_iter=2,
                        resume_global_step_from=0,
                        is_inception=False):

    # train_from_epoch, save_each_iter, resume_global_step_from,
    
    since = time.time()

    # meters
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # manca il writer
    writer = SummaryWriter(os.path.join('logs', model_name))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    dataloaders = {
        'train': dst_container.training_loader,
        'validation': dst_container.validation_loader
    }

    global_step = 0 + resume_global_step_from
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # print("n di elementi del training", inputs.shape[0])
                global_step += inputs.shape[0]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # loggiamo i risultati iterazione per iterazione solo durante il training
                if phase == 'train':
                    writer.add_scalar('loss/train', running_loss , global_step=global_step)
                    writer.add_scalar('accuracy/train', running_corrects , global_step=global_step)


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # una volta finita l'epoca sia nel caso di training che di test loggiamo le stime finali
            writer.add_scalar('loss/' + phase, epoch_loss, global_step=global_step)
            writer.add_scalar('accuracy/' + phase, epoch_acc , global_step=global_step)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'validation':
                val_acc_history.append(epoch_acc)
        

        # if save_on_runtime is True:
        if ((epoch+1) % save_each_iter == 0):
            torch.save(model.state_dict(), 'models/%s-%d.pth'%(model_name, (epoch+1)+train_from_epoch ))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history