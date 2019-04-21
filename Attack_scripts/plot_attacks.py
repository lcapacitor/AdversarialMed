import os
import util
import torch
import argparse
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable



CKPT_PATH = 'models/pneu_model_aug.ckpt'
IMG_PATH_NORM = './img/0/'
IMG_PATH_PNEU = './img/1/'
BI_ClASS_NAMES = ['Normal', 'Pneumonia']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def attackPlots(epsilon, fileFolder, num_iters, alpha, is_plot, attack):

    # Define loss function
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Load chexnet model
    CheXnet_model = util.loadPneuModel(CKPT_PATH)

    files = os.listdir(fileFolder)

    if fileFolder.split('/')[-1] == '':
        label = int(fileFolder.split('/')[-2])
    else:
        label = int(fileFolder.split('/')[-1])

    label_var = Variable(torch.Tensor([float(label)]).long(), requires_grad=False).to(device)

    for f in files:
        # Get images
        _, _, img_ts = util.singleImgPreProc(os.path.join(fileFolder, f))
        img_ts.requires_grad = True

        # Predict with model
        output = CheXnet_model(img_ts)
        _, preds = torch.max(output, 1)
        scores = output.data.cpu().numpy()
        print ('Actual prediction: {}, probs: {}, GT label: {}'.format(BI_ClASS_NAMES[preds], scores, BI_ClASS_NAMES[label]))
        loss = loss_fn(output, label_var)
        loss.backward()
        
        adv_img = util.generateAdvExamples(CheXnet_model, loss_fn, label_var, img_ts, epsilon, num_iters, alpha, attack)

        # Predict with adversarial
        f_ouput = CheXnet_model(adv_img)
        _, f_preds = torch.max(f_ouput, 1)
        f_scores = f_ouput.data.cpu().numpy()
        print ('Adv prediction: {}, probs: {}'.format(BI_ClASS_NAMES[f_preds], f_scores))

        # Plot results
        if is_plot:
            perturb = (adv_img - img_ts) / epsilon
            util.plotFigures(img_ts, preds, scores, adv_img, f_preds, f_scores, perturb, epsilon)


def main(args):
    attack = args.attack
    epsilon = args.epsilon
    num_iters = args.num_iters
    alpha = args.alpha
    is_plot = args.is_plot
    fileFolder = args.fpath
    attackPlots(epsilon, fileFolder, num_iters, alpha, is_plot, attack)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', type=str, required=True, 
        help='specify which attack to use: fgsm, i-fgsm, pgd, pixel, mi-fgsm')
    parser.add_argument('--fpath', type=str, required=True, help='path to the images to attack, e.g. img/0')
    parser.add_argument('--epsilon', type=float, default=0.02, help='A float value of epsilon, default is 0.05')
    parser.add_argument('--num_iters', type=int, default=10, help='Number of iterations, default is 10')
    parser.add_argument('--alpha', type=float, default=0.005, help='A float value of alpha, default is 0.005')
    parser.add_argument('--is_plot', type=int, default=1, help='if plot the attack results, 1: True, 0: False')

    args = parser.parse_args()
    main(args)