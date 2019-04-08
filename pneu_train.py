import os
import re
import copy
import time
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix


CKPT_PATH = 'chexnet_model.pth.tar'
N_CLASSES = 2
CLASS_NAMES = ['Pneumonia']
DATA_DIR = 'D:/Professional/Research/Kaggle Competitions/RSNA Pneumonia Detection Challenge/data/chest_xray_pneu_data/'
BATCH_SIZE = 16


def trained_chexnet(checkpoint_path):
    '''
    Load the trained ChexNet with the given checkpoint directory
    Input:
        checkpoint_path:    (str) the directory to the trained ChexNet model
    Return:
        the loaded ChexNet model
    '''
    # construct ChexNet structure
    out_size =14
    model = torchvision.models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, out_size),
        nn.Softmax(1)
    )
    # change the names in Densenet old version
    pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    # load checkpoint
    if torch.cuda.is_available():
        modelCheckpoint = torch.load(checkpoint_path)
    else:
        modelCheckpoint = torch.load(checkpoint_path, map_location='cpu')
    # load weights from the loaded checkpoint
    state_dict = modelCheckpoint['state_dict']
    # replacement from the old version
    for key in list(state_dict.keys()):
        old_key = key
        key = key.replace('module.densenet121.', '')
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
        else:
            new_key = key
        state_dict[new_key] = state_dict[old_key]
        del state_dict[old_key]

    model.load_state_dict(state_dict)
    return model


def train():

    num_epochs = 20
    lr = 0.0001
    weight_decay = 1e-5
    nnIsTrained = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = trained_chexnet(CKPT_PATH)
    model.classifier = nn.Sequential(
        nn.Linear(1024, N_CLASSES),
        nn.Softmax(1)
    )
    model = model.to(device)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
                   'val':   torch.utils.data.DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    #loss_fn = nn.BCELoss()
    loss_fn = nn.CrossEntropyLoss()
    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    # define lr_scheduler: if the loss is not improved for 3 epoch, then reduce lr by 2
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, threshold=0.0001)

    print ("train_model(): Start training...")
    since = time.time() # timing
    best_loss = 999.9
    best_model_wts = copy.deepcopy(model.state_dict())
    best_C = np.zeros((2,2))
    count = 0

    for epoch in range(num_epochs):
        if count > 3:
            break

        if optimizer.param_groups[0]['lr'] < 1e-7:
            count = count + 1

        print('\nEpoch {}/{}, lr: {}, wd: {}'.format(epoch + 1, num_epochs,
                            optimizer.param_groups[0]['lr'], weight_decay))

        print('-' * 50)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            C = np.zeros((2,2))

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.type(torch.LongTensor).to(device)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                #print (preds_np, labels_np)
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'val':
                    C += confusion_matrix(labels.data.cpu().numpy(), preds.cpu().numpy(), labels=[0,1])

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # step learning rate scheduler
            if phase == 'val':
                scheduler.step(epoch_loss)
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_C = C
                print (best_C)

    # Finish training, printing out results
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:.4f}\nBest C:'.format(best_loss, best_C))

    # save best model weights
    torch.save(best_model_wts, 'pneu_model.ckpt')


if __name__ == '__main__':
    train()
