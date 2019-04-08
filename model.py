# encoding: utf-8

"""
The main CheXNet model implementation.
"""


import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score


CKPT_PATH = 'model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = './img'
TEST_IMAGE_LIST = './img/list.txt'
BATCH_SIZE = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def load_weights(path):
    '''
    Load trained weights from the given path.
    If the current machine is CPU only, then read with map CPU.
    '''
    if torch.cuda.is_available():
        weights = torch.load(path)
    else:
        weights = torch.load(path, map_location='cpu')
    return weights


def convert_state_dict_keys(state_dict):

	pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

	for key in list(state_dict.keys()):
		res = pattern.match(key)
		if res:
			new_key = res.group(1) + res.group(2)
			state_dict[new_key] = state_dict[key]
			del state_dict[key]

	return state_dict


def main():

    cudnn.benchmark = True

    # initialize and load the model
    model = DenseNet121(N_CLASSES).to(device)
    model = torch.nn.DataParallel(model).to(device)

    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = load_weights(CKPT_PATH)
        state_dict = convert_state_dict_keys(checkpoint['state_dict'])
        model.load_state_dict(state_dict)
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TEST_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)

    # switch to evaluate mode
    model.eval()

    for i, (inp, target) in enumerate(test_loader):
        target = target.to(device)
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = torch.autograd.Variable(inp.view(-1, c, h, w).to(device), volatile=True)
        output = model(input_var)
        output_mean = output.view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output_mean.data), 0)

        print (output_mean)

    #AUROCs = compute_AUCs(gt, pred)
    #AUROC_avg = np.array(AUROCs).mean()
    #print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    #for i in range(N_CLASSES):
    #    print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


if __name__ == '__main__':
    main()