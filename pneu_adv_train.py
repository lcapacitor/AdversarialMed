import os
import re
import copy
import time
import util
import numpy as np
import torch
import argparse
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix


CKPT_PATH = './Attack_scripts/models/model_14_class.pth.tar'
N_CLASSES = 2
CLASS_NAMES = ['Pneumonia']
DATA_DIR = '../../chex_pneu_data_aug/'
BATCH_SIZE = 20


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


def train(attack_type):

    num_epochs = 20
    lr = 0.0001
    weight_decay = 1e-5
    nnIsTrained = True

    # Adv training params
    alp = 0.7

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
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
                   'val':   torch.utils.data.DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=2)}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    #loss_fn = nn.BCELoss()
    loss_fn = nn.CrossEntropyLoss()
    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    # define lr_scheduler: if the loss is not improved for 3 epoch, then reduce lr by 2
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4, threshold=0.0001)

    print ("train_model(): Start training...")
    since = time.time() # timing
    best_loss = 999.9
    best_model_wts = copy.deepcopy(model.state_dict())
    best_C = np.zeros((2,2))
    best_C_adv = np.zeros((2,2))
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
            running_corrects_adv = 0
            C = np.zeros((2,2))
            C_adv = np.zeros((2,2))

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc='{} iters'.format(phase), leave=False):
                inputs = inputs.to(device)
                inputs.requires_grad = True
                labels = labels.type(torch.LongTensor).to(device)
                # forward
                # track history if only in train
                # with torch.set_grad_enabled(phase == 'train'):
            	# Clean training and clean loss
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss_clean = loss_fn(outputs, labels)

                # Adv training and adv loss with FGSM
                loss_clean.backward(retain_graph=True)

                loss_adv = []
                for att in attack_type:
                    adv_exams = util.generateAdvExamples(model, loss_fn, labels, inputs, epsilon=0.02, num_iters=10, alpha=0.005, attack_type=att)
                    outputs_adv = model(adv_exams)
                    _, preds_adv = torch.max(outputs_adv, 1)
                    loss_adv.append(loss_fn(outputs_adv, preds.to(device)))	# train adv with predicted label to prevent label leaking
                loss_adv_mean = torch.mean(torch.stack(loss_adv))

                # total loss
                loss = alp * loss_clean + (1 - alp) * loss_adv_mean

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
                running_corrects_adv += torch.sum(preds_adv == labels.data)
                if phase == 'val':
                    C += confusion_matrix(labels.data.cpu().numpy(), preds.cpu().numpy(), labels=[0,1])
                    C_adv += confusion_matrix(labels.data.cpu().numpy(), preds_adv.cpu().numpy(), labels=[0,1])

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_acc_adv = running_corrects_adv.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Clean Acc: {:.4f} Adv Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_acc_adv))
            # step learning rate scheduler
            if phase == 'val':
                scheduler.step(epoch_loss)
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_C = C
                best_C_adv = C_adv
                print (C)
                print (C_adv)
            # save checkpoints
            if phase == 'val' and (epoch+1) % 5 == 0 and epoch > 0:
                ckpt_model = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, './models/checkpoints/pneu_adv_model_{}_{}_e{}.ckpt'
                    .format('_'.join(attack_type), alp, epoch+1))

    # Finish training, printing out results
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:.4f}\nBest C:{}\nBest C_adv:{}'.format(best_loss, best_C, best_C_adv))

    # save best model weights
    torch.save(best_model_wts, './models/pneu_adv_model_{}_{}.ckpt'.format('_'.join(attack_type), alp))


def main(args):
    attack_type = args.attack
    train(attack_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', type=str, nargs='+', required=True, 
        help='specify which attack to use: fgsm, i-fgsm, pgd, pixel, mi-fgsm')
    args = parser.parse_args()
    main(args)
