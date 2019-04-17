import os
import cv2
import copy
import math
import time
import torch
import joblib
import random
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from vae_conv_patch import VAE
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score


PNEU_PATH = '../models/pneu_model.ckpt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]


def imgProc(img_path, is_target):
    ori_img = cv2.imread(img_path)
    ori_img = cv2.resize(ori_img, (224, 224))
    img = ori_img.copy().astype(np.float32)

    img /= 255.0
    if not is_target:
        img = (img - mean)/std
    img = img.transpose(2, 0, 1)
    img_ts = Variable(torch.from_numpy(img).type(torch.float)).to(device)

    return img_ts


class ImageDataset(Dataset):
    '''
    clean_dir:
        (str) path of the root folder of all clean images
    adv_dir:
        (str) path to the root folder of all attacked images
    attack_type:
        (str) type of attack name
    transform:
        (transforms) transforms for data
    is_train:
        (bool) if the data for training or test
    '''
    def __init__(self, clean_dir, adv_dir, attack_type, setName, data_type):
        self.ori_fileList = []
        self.adv_fileList = []
        self.data_type = data_type
        ori_folder = os.path.join(clean_dir, setName)
        adv_folder = os.path.join(adv_dir, attack_type, setName)
        for path, subdirs, files in os.walk(ori_folder):
            for f in files:
                self.ori_fileList.append(os.path.join(path, f))
                adv_fname = 'adv_' + f
                self.adv_fileList.append(os.path.join(adv_folder, adv_fname))
        assert len(self.ori_fileList)==len(self.adv_fileList)

    def __len__(self):
        return len(self.ori_fileList)

    def __getitem__(self, index):
        if self.data_type == 'clean':
            x = imgProc(self.ori_fileList[index], is_target=False)
        else:
            x = imgProc(self.adv_fileList[index], is_target=False)
            
        y = int(self.ori_fileList[index].split('/')[-2])
        label = torch.LongTensor([y])
        # Assertions
        _adv = '_'.join(self.adv_fileList[index].split('/')[-1].split('_')[1:])
        _ori = self.ori_fileList[index].split('/')[-1]
        assert (_adv == _ori)
        return x, label


class classifier(nn.Module):
    def __init__(self, input_size, hidd_size, out_size):
        super(classifier, self).__init__()

        self.input_size = input_size
        self.hidd_size = hidd_size
        self.linearBlock1 = nn.Sequential(
                nn.Linear(input_size, hidd_size*2),
                nn.ReLU(),
            )
        self.linearBlock2 = nn.Sequential(
                nn.Linear(hidd_size*2, hidd_size*2),
                nn.ReLU(),
            )
        self.linearBlock3 = nn.Sequential(
                nn.Linear(hidd_size*2, out_size),
                nn.Sigmoid(),
            )

    def forward(self, x):
        out = self.linearBlock1(x)
        out = self.linearBlock2(out)
        out = self.linearBlock3(out)
        return out


def loadVAEmodel(model_path):
    z_size = 2048
    hidden_dim = 64
    drop_p = 0.5
    image_size = 224
    patch_size = 32
    patch_stride = image_size // patch_size
    channel_num = 3
    is_res = True
    model = VAE(patch_size, channel_num, hidden_dim, z_size, is_res, drop_p)

    # load checkpoint
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model.eval().to(device)


def loadPneuModel(model_path):
    out_size = 2
    model = torchvision.models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, out_size),
        nn.Softmax(1)
    )

    # load checkpoint
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location='cpu')

    model.load_state_dict(state_dict)
    return model.eval().to(device)


def trainClassifier(clean_dir, adv_dir, attack_type):

    lr = 1e-3
    weight_decay = 1e-5
    batch_size = 64
    num_epochs = 20
    best_loss = math.inf
    best_C = np.zeros((2,2))
    input_size = 2048
    hidd_size = 2048
    out_size = 1
    model_path = './trained_weights/vae_patch32_fgsm_zdim2048_hdim64_e50_lr0005.torch'
    data_type = 'clean'

    dataset = {x: ImageDataset(clean_dir, adv_dir, attack_type, x, data_type) for x in ['train', 'val']}
    dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val']}
    dataloaders = {'train': DataLoader(dataset['train'], batch_size=batch_size, shuffle=True,  num_workers=0),
                   'val'  : DataLoader(dataset['val'],   batch_size=batch_size, shuffle=False, num_workers=0)}
    encode_model = loadVAEmodel(model_path)
    model = classifier(input_size, hidd_size, out_size).to(device)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, threshold=1e-7)

    # test
    for epoch in range(num_epochs):
        print('\nEpoch {}/{}, lr: {}, wd: {}'.format(epoch + 1, num_epochs,
              optimizer.param_groups[0]['lr'], weight_decay))
        print('-' * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            C = np.zeros((2,2))

            for inputs, targets in tqdm(dataloaders[phase], desc='test iters', leave=False):
                inputs = inputs.to(device)
                targets = targets.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    for j in range(7):
                        for i in range(7):
                            row_s = 32 * j
                            row_e = row_s + 32
                            col_s = 32 * i
                            col_e = col_s + 32

                            encoded = encode_model.encoder(inputs[:,:,row_s:row_e,col_s:col_e])
                            feature_volume = encoded.size(1) * encoded.size(2) * encoded.size(3)
                            unrolled = encoded.view(-1, feature_volume)
                            z_mean   = encode_model.q_mean(unrolled)
                            z_logvar = encode_model.q_logvar(unrolled)

                            std = z_logvar.mul(0.5).exp_()
                            eps = (
                                Variable(torch.randn(std.size())).to(device)
                            )
                            z = eps.mul(std).add_(z_mean)

                            outputs = model(z)
                            preds = outputs > 0.5
                            loss = loss_fn(outputs, targets)
                            if phase == 'train':
                                # zero the parameter gradients
                                optimizer.zero_grad()
                                # backward-prop
                                loss.backward()
                                optimizer.step()
                            running_loss += loss.item() * inputs.size(0)
                            running_corrects += torch.sum(preds == targets.type(torch.uint8).data)
                            C += confusion_matrix(targets.data.cpu().numpy(), preds.cpu().numpy(), labels=[0,1])

            epoch_loss = running_loss / (dataset_sizes[phase] * 49)
            epoch_acc = running_corrects.double() / (dataset_sizes[phase] * 49)

            # Output training/val results
            print('{} Loss: total: {:.4f}, epoch_acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_C = C
                print (C)


def testVAEDefense(clean_dir, adv_dir, attack_type):

    lr = 1e-3
    weight_decay = 1e-5
    batch_size = 20

    model_path = './trained_weights/vae_patch32_fgsm_zdim2048_hdim64_e50_lr0005.torch'
    data_type = ['clean']
    setName = 'val'
    pred_probs_vae = {x: [] for x in data_type}
    running_corrects_vae = {x: 0 for x in data_type}
    pred_probs = {x: [] for x in data_type}
    running_corrects = {x: 0 for x in data_type}
    

    datasets    = {x: ImageDataset(clean_dir, adv_dir, attack_type, setName, x) for x in data_type}
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in data_type}

    vaeModel = loadVAEmodel(model_path)
    pneuModel = loadPneuModel(PNEU_PATH)
    plt.figure(figsize=(10, 8))

    for dt in data_type:
        gt_labels = []
        for inputs, labels in tqdm(dataloaders[dt], desc='{} iters'.format(dt), leave=False):
            gt_labels += labels.tolist()
            labels = labels.squeeze(1)
            rec_inputs = torch.zeros((inputs.shape))
            # reconstruct input from parts
            for j in range(7):
                for i in range(7):
                    row_s = 32 * j
                    row_e = row_s + 32
                    col_s = 32 * i
                    col_e = col_s + 32

                    _, rec_parts = vaeModel(inputs[:,:,row_s:row_e,col_s:col_e])
                    rec_inputs[:,:,row_s:row_e,col_s:col_e] = rec_parts

            # Show rec images
            #rec = rec_inputs[0].data.cpu().numpy()
            #rec = np.transpose(rec, (1,2,0))
            #rec = np.clip(rec, 0, 1)
            #plt.imshow(rec)
            #plt.show()
            
            # Normalize rec_inputs
            mean_ts = torch.Tensor(mean).view(1, 3, 1, 1)
            std_ts  = torch.Tensor(std). view(1, 3, 1, 1)
            rec_inputs = (rec_inputs - mean_ts) / std_ts

            # Predict with pneuModel
            outputs = pneuModel(inputs)
            _, preds = torch.max(outputs, 1)
            pred_probs[dt] += outputs[:, 1].tolist()
            running_corrects[dt] += torch.sum(preds.cpu() == labels.data).item()
            
            # Predict with VAErec
            rec_ouputs = pneuModel(rec_inputs.to(device))
            _, rec_preds = torch.max(rec_ouputs, 1)
            pred_probs_vae[dt] += rec_ouputs[:, 1].tolist()
            running_corrects_vae[dt] += torch.sum(rec_preds.cpu() == labels.data).item()


        print (len(gt_labels), len(pred_probs_vae[dt]))
        assert len(gt_labels)==len(pred_probs_vae[dt])
        auc = roc_auc_score(gt_labels, pred_probs[dt])
        fpr, tpr, thresholds = roc_curve(gt_labels, pred_probs[dt])
        accuracy = running_corrects[dt] / len(datasets[dt])
        print('pneuModel on {} data: Accuracy: {:.4f}, AUC: {:.4f}'.format(dt, accuracy, auc))
        plt.plot(fpr, tpr, '-.', label='pneuModel on {} (auc = {:.4f})'.format(dt, auc))

        auc = roc_auc_score(gt_labels, pred_probs_vae[dt])
        fpr, tpr, thresholds = roc_curve(gt_labels, pred_probs_vae[dt])
        accuracy = running_corrects_vae[dt] / len(datasets[dt])
        print('VAErec on {} data: Accuracy: {:.4f}, AUC: {:.4f}'.format(dt, accuracy, auc))
        plt.plot(fpr, tpr, '-.', label='VAErec on {} (auc = {:.4f})'.format(dt, auc))

    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.legend()
    plt.show()



def main(args):
    clean_dir = args.clean_dir
    adv_dir = args.adv_dir
    attack_type = args.attack_type
    testVAEDefense(clean_dir, adv_dir, attack_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_dir', type=str, required=True, help='root folder of the clean images')
    parser.add_argument('--adv_dir', type=str, required=True, help='root folder of the adversarial images')
    parser.add_argument('--attack_type', type=str, required=True, help='type of attacks, e.g. fgsm, b_iter')
    args = parser.parse_args()
    main(args)


