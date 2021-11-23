# Import libraries
import torch
import time
import os
from torch import nn
import torch.optim as optim
import copy
from torch.optim import lr_scheduler
import wandb


def train_model(num_exp, model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    print(f'device: {device}')
    model = model.to(device)
    
    wandb.watch(model)
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            
            model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'train':
                wandb.log({"epoch": epoch, 'train_accuracy': epoch_acc, 'train_loss': epoch_loss})
            else:
                wandb.log({"epoch": epoch, 'val_accuracy': epoch_acc, 'val_loss': epoch_loss})

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),'../runs/exp_' + str(num_exp) + '/model.pt')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # wandb.log({"best_val_acc": best_acc})

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def define_hyperparameters(device, model, lr=0.001, momentum=0.9, scheduler_step_size=10, scheduler_gamma=0.1):
    criterion = nn.CrossEntropyLoss().to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=scheduler_step_size, gamma=scheduler_gamma)
    
    return criterion, optimizer_ft, exp_lr_scheduler


def train(experiment_num, seed, model, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, device, num_epochs=100):
    
    # Seed
    torch.manual_seed(seed)
    
    exp_num = experiment_num
    if os.path.isdir('../runs/exp_' + str(exp_num)) == False:
        os.mkdir('../runs/exp_' + str(exp_num))
    model_exp1 = train_model(exp_num, model, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs, device=device)
