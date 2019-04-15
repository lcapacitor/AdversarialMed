import os
import util
import torch
import argparse
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, roc_auc_score

from advertorch.defenses import MedianSmoothing2D
from advertorch.defenses import BitSqueezing
from advertorch.defenses import JPEGFilter


_to_pil_image = transforms.ToPILImage()
_to_tensor = transforms.ToTensor()

PNEU_PATH = 'models/pneu_model.ckpt'
PNEU_ADV  = 'models/pneu_adv_model.ckpt'
CHEX_PATH = 'models/model_14_class.pth.tar'
BI_ClASS_NAMES = ['Normal', 'Pneumonia']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 20


def testPerformance(attack_type, epsilon, data_path, model_type, defense_type):
    # Initialize Defense
    bits_squeezing = BitSqueezing(bit_depth=5)
    median_filter = MedianSmoothing2D(kernel_size=3)
    jpeg_filter = JPEGFilter(23)

    defense_jpeg = nn.Sequential(
        jpeg_filter,
        # bits_squeezing,
        # median_filter,
    )

    # Load chexnet model
    if model_type == 'pneu':
        model = util.loadPneuModel(PNEU_PATH)
    if model_type == 'chex':
        model = util.loadChexnet14(CHEX_PATH)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Setup data loader
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(util.mean, util.std),
    ])
    image_dataset = datasets.ImageFolder(data_path, data_transform)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    running_corrects = 0
    pred_probs = []
    gt_labels = []

    # Intialize the error container for attacks
    preds_attacks = {}
    for att in attack_type:
        preds_attacks[att] = {'pred_probs_adv': [], 'running_corrects_adv':0}
        # Intialize the error container for defenses
        if defense_type is not None:
            for defs in defense_type:
                preds_attacks[att][defs] = {'pred_probs_defense':[], 'pred_probs_adv_defense':[],
                                            'running_corrects_defense': 0, 'running_corrects_adv_defense': 0}

    for inputs, labels in tqdm(dataloader, desc='test iters', leave=False):
        inputs = inputs.to(device)
        inputs.requires_grad = True

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        pred_probs += outputs[:, 1].tolist()
        gt_labels += labels.tolist()
        running_corrects += torch.sum(preds.cpu() == labels.data).item()

        loss = loss_fn(outputs, labels.to(device))
        loss.backward()

        for att in attack_type:
            num_iters = 10
            alpha = 0.005
            adv_img = util.generateAdvExamples(model, loss_fn, labels.to(device), inputs, epsilon, num_iters, alpha, att)

            # Predict with adversarial
            f_ouput = model(adv_img)
            _, f_preds = torch.max(f_ouput, 1)
            preds_attacks[att]['pred_probs_adv'] += f_ouput[:, 1].tolist()
            preds_attacks[att]['running_corrects_adv'] += torch.sum(f_preds.cpu() == labels.data).item()

            if defense_type is not None:
                # Loop over each defense
                for defs in defense_type:
                    if defs.lower() == "jpeg":
                        # reconstruct for defense
                        reconstruct_inputs_clean = util.unpreprocessBatchImages(inputs).permute(0, 3, 1 ,2)
                        reconstruct_inputs_adv = util.unpreprocessBatchImages(adv_img).permute(0, 3, 1 ,2)
                        defense_input = defense_jpeg(reconstruct_inputs_clean)
                        defense_adv = defense_jpeg(reconstruct_inputs_adv)

                        # propceoss the images again for pretrained model
                        lst_img = []
                        for img in defense_input:
                            img = _to_pil_image(img.detach().clone().cpu())
                            lst_img.append(data_transform(img))
                        defense_input = torch.stack(lst_img).to(device)

                        lst_img = []
                        for img in defense_adv:
                            img = _to_pil_image(img.detach().clone().cpu())
                            lst_img.append(data_transform(img))
                        defense_adv = torch.stack(lst_img).to(device)
                        model_defs = model

                    elif defs.lower() == "adv_train":
                        model_defs = util.loadPneuModel(PNEU_ADV)
                        defense_input = inputs
                        defense_adv = adv_img

                    else:
                        raise AttributeError("Provided defense type not supported")

                    # Predict with defensed images
                    defense_outputs = model_defs(defense_input)
                    _, defense_preds = torch.max(defense_outputs, 1)
                    preds_attacks[att][defs]['pred_probs_defense'] += defense_outputs[:, 1].tolist()
                    preds_attacks[att][defs]['running_corrects_defense'] += torch.sum(defense_preds.cpu() == labels.data).item()

                    f_ouput_defense = model_defs(defense_adv)
                    _, f_preds_defense = torch.max(f_ouput_defense, 1)
                    preds_attacks[att][defs]['pred_probs_adv_defense'] += f_ouput_defense[:, 1].tolist()
                    preds_attacks[att][defs]['running_corrects_adv_defense'] += torch.sum(f_preds_defense.cpu() == labels.data).item()

            # debug plot out images
            # plotCleanAdversariallDefenseImages(inputs, adv_img, defense_input, defense_adv)

    # compute metrices on clean data and plot roc-auc
    auc = roc_auc_score(gt_labels, pred_probs)
    fpr, tpr, thresholds = roc_curve(gt_labels, pred_probs)
    accuracy = running_corrects / len(image_dataset)
    print('Clean Examples: Accuracy: {:.4f}, AUC: {:.4f}'.format(accuracy, auc))
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, '-.', label='clean (auc = {:.4f})'.format(auc))

    # compute metrices on adv with all attacks and plot roc-auc
    for att in attack_type:
        auc_adv = roc_auc_score(gt_labels, preds_attacks[att]['pred_probs_adv'])
        fpr_adv, tpr_adv, thresholds_adv = roc_curve(gt_labels, preds_attacks[att]['pred_probs_adv'])
        accuracy_adv = preds_attacks[att]['running_corrects_adv'] / len(image_dataset)
        print('{} Adversarial Examples: Accuracy: {:.4f}, AUC: {:.4f}'.format(att, accuracy_adv, auc_adv))
        plt.plot(fpr_adv, tpr_adv, '-.', label='{} adversarial (auc = {:.4f})'.format(att, auc_adv))

        if defense_type is not None:
            for defs in defense_type:
                auc_defense = roc_auc_score(gt_labels, preds_attacks[att][defs]['pred_probs_defense'])
                auc_adv_defense = roc_auc_score(gt_labels, preds_attacks[att][defs]['pred_probs_adv_defense'])

                fpr_defense, tpr_defense, thresholds_defense = roc_curve(gt_labels, preds_attacks[att][defs]['pred_probs_defense'])
                fpr_adv_defense, tpr_adv_defense, thresholds_adv_defense = roc_curve(gt_labels, preds_attacks[att][defs]['pred_probs_adv_defense'])

                accuracy_defense = preds_attacks[att][defs]['running_corrects_defense'] / len(image_dataset)
                accuracy_adv_defense = preds_attacks[att][defs]['running_corrects_adv_defense'] / len(image_dataset)

                print('{} defense on Clean Examples: Accuracy: {:.4f}, AUC: {:.4f}'.format(defs, accuracy_defense, auc_defense))
                print('{} defense on Adversarial Examples: Accuracy: {:.4f}, AUC: {:.4f}'.format(defs, accuracy_adv_defense,
                                                                                              auc_adv_defense))

                plt.plot(fpr_defense, tpr_defense, '-.', label='{} defense on clean (auc = {:.4f})'.format(defs, auc_defense))
                plt.plot(fpr_adv_defense, tpr_adv_defense, '-.',
                         label='{} defense on {} adversarial (auc = {:.4f})'.format(defs, att, auc_adv_defense))

    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.legend()
    plt.show()


def main(args):
    attack_type  = args.attack
    epsilon      = args.epsilon
    data_path    = args.path
    model_type   = args.model
    defense_type = args.defense
    testPerformance(attack_type, epsilon, data_path, model_type, defense_type)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', type=str, nargs='+', required=True, help='specify which attack to use: fgsm and/or b_iter, default is fgsm')
    parser.add_argument('--epsilon', type=float, default=0.02, help='a float value of epsilon, default is 0.02')
    parser.add_argument('--path', type=str, required=True, help='Path to the test images')
    parser.add_argument('--model', type=str, default='pneu',
                        help='specify which model will be tested: pneu or chex, default is pneu')
    parser.add_argument('--defense', type=str, nargs='+', default=None, help='specify which defense to use: JPEG, default is None')
    args = parser.parse_args()
    main(args)
