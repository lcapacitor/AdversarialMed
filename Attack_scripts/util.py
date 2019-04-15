import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable



BI_ClASS_NAMES = ['Normal', 'Pneumonia']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]


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


def loadChexnet14(checkpoint_path):
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
    return model.eval().to(device)


def singleImgPreProc(img_path):

	ori_img = cv2.imread(img_path)
	ori_img = cv2.resize(ori_img, (224, 224))
	img = ori_img.copy().astype(np.float32)

	img /= 255.0
	img = (img - mean)/std
	img = img.transpose(2, 0, 1)

	img_ts = Variable(torch.from_numpy(img).type(torch.float).unsqueeze(0)).to(device)
	img_ts.requires_grad = True
	#img_ts = Variable(torch.from_numpy(img).type(torch.float).unsqueeze(0), requires_grad=True).to(device)
	return ori_img, img, img_ts


def saveImage(save_folder, ori_fname, adv_img):

	if not os.path.isdir(save_folder):
		print ('The given save folder does not exists, creat ./adv_img/ instead')
		os.mkdir('adv_img/')
		save_folder = 'adv_img/'
	
	x_adv = adv_img.squeeze(0).data.cpu()
	x_adv = x_adv.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3,1,1)).detach().numpy()
	x_adv = np.transpose(x_adv, (1,2,0))   # C X H X W  ==>   H X W X C
	x_adv = np.clip(x_adv, 0, 1)
	x_adv = x_adv * 255
	print (np.max(x_adv), np.min(x_adv))

	adv_fname = 'adv_' + ori_fname
	save_path = os.path.join(save_folder, adv_fname)

	print ('save to: ', save_path)
	cv2.imwrite(save_path, x_adv)

'''
This function take a batch of pre-processed images array and reconstruct the corresponding original images
'''
def unpreprocessBatchImages(batch_imgs):
	batch_size = batch_imgs.shape[0]
	batch_imgs = batch_imgs.data.cpu()
	batch_imgs = batch_imgs.mul(torch.FloatTensor(std).view(3, 1, 1).expand(batch_size, 3, 1, 1)).add(torch.FloatTensor(mean).view(3, 1, 1).expand(batch_size, 3, 1, 1)).detach().numpy()
	batch_imgs = np.transpose(batch_imgs, (0, 2, 3, 1))  # C X H X W  ==>   H X W X C
	return torch.from_numpy(np.clip(batch_imgs, 0, 1))



def plotCleanAdversariallDefenseImages(org, adv, defense_clean, defense_adversarial):
	# x = org.squeeze(0).data.cpu()
	# x = x.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3, 1, 1)).detach().numpy()
	# x = np.transpose(x, (1, 2, 0))  # C X H X W  ==>   H X W X C
	# x = np.clip(x, 0, 1)
	x = unpreprocessBatchImages(org)[0]

	# x_adv = adv.squeeze(0).data.cpu()
	# x_adv = x_adv.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3, 1, 1)).detach().numpy()
	# x_adv = np.transpose(x_adv, (1, 2, 0))  # C X H X W  ==>   H X W X C
	# x_adv = np.clip(x_adv, 0, 1)
	x_adv = unpreprocessBatchImages(adv)[0]

	# x_clean_defense = defense_clean.squeeze(0).data.cpu()
	# x_clean_defense = x_clean_defense.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3, 1, 1)).detach().numpy()
	# x_clean_defense = np.transpose(x_clean_defense, (1, 2, 0))  # C X H X W  ==>   H X W X C
	# x_clean_defense = np.clip(x_clean_defense, 0, 1)
	x_clean_defense = unpreprocessBatchImages(defense_clean)[0]

	# x_adv_defense = defense_adversarial.squeeze(0).data.cpu()
	# x_adv_defense = x_adv_defense.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3, 1, 1)).detach().numpy()
	# x_adv_defense = np.transpose(x_adv_defense, (1, 2, 0))  # C X H X W  ==>   H X W X C
	# x_adv_defense = np.clip(x_adv_defense, 0, 1)
	x_adv_defense = unpreprocessBatchImages(defense_adversarial)[0]


	figure, ax = plt.subplots(1, 4, figsize=(18, 8))

	ax[0].imshow(x)
	ax[0].set_title('Clean Example', fontsize=20)

	ax[1].imshow(x_adv)
	ax[1].set_title('Adversarial Example', fontsize=20)

	ax[2].imshow(x_clean_defense)
	ax[2].set_title('Defense Clean Example', fontsize=20)

	ax[3].imshow(x_adv_defense)
	ax[3].set_title('Defense Adversarial Example', fontsize=20)

	plt.show()



def plotFigures(oriImg, preds, ori_score, advImg, f_preds, f_score, x_grad, epsilon):

	# Get original image
	x = oriImg.squeeze(0).data.cpu()
	x = x.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3,1,1)).detach().numpy()
	x = np.transpose(x, (1,2,0))   # C X H X W  ==>   H X W X C
	x = np.clip(x, 0, 1)

    	# Get adversarail image
	x_adv = advImg.squeeze(0).data.cpu()
	x_adv = x_adv.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3,1,1)).detach().numpy()
	x_adv = np.transpose(x_adv, (1,2,0))   # C X H X W  ==>   H X W X C
	x_adv = np.clip(x_adv, 0, 1)

	# Get perturbation image
	x_grad = x_grad.squeeze(0).data.cpu().numpy()
	x_grad = np.transpose(x_grad, (1, 2, 0))
	x_grad = np.clip(x_grad, 0, 1)

	# Plot images
	figure, ax = plt.subplots(1,3, figsize=(18,8))
	ax[0].imshow(x)
	ax[0].set_title('Clean Example', fontsize=20)

	ax[1].imshow(x_grad)
	ax[1].set_title('Perturbation', fontsize=20)
	ax[1].set_yticklabels([])
	ax[1].set_xticklabels([])
	ax[1].set_xticks([])
	ax[1].set_yticks([])

	ax[2].imshow(x_adv)
	ax[2].set_title('Adversarial Example', fontsize=20)

	ax[0].axis('off')
	ax[2].axis('off')

	ax[0].text(1.1,0.5, "+{}*".format(round(epsilon,3)), size=15, ha="center",
             transform=ax[0].transAxes)

	ax[0].text(0.5,-0.13, "Prediction: {}\n Probability: {:.4f}".format(BI_ClASS_NAMES[preds], np.max(ori_score)), size=15, ha="center", transform=ax[0].transAxes)

	ax[1].text(1.1,0.5, " = ", size=15, ha="center", transform=ax[1].transAxes)

	ax[2].text(0.5,-0.13, "Prediction: {}\n Probability: {:.4f}".format(BI_ClASS_NAMES[f_preds], np.max(f_score)), size=15, ha="center", transform=ax[2].transAxes)

	plt.show()


def generateAdvExamples(model, loss_fn, label_var, inputs, epsilon, num_iters, alpha, attack_type):
    if attack_type.lower() == 'fgsm':
        # FGSM get adversarial
        x_grad = torch.sign(inputs.grad.data)
        perturbation = epsilon * x_grad
        x_adv = inputs.data + perturbation
        return x_adv

    elif attack_type.lower() == 'b_iter':
        x_ori = Variable(inputs.data, requires_grad=True)
        x_adv = Variable(inputs.data, requires_grad=True)
        # Loop over each iteration
        for i in range(num_iters):
            zero_gradients(x_adv)
            output_iter = model(x_adv)
            loss_iter = loss_fn(output_iter, label_var)
            loss_iter.backward()

            x_grad  = alpha * torch.sign(x_adv.grad.data)

            perturbation = (x_adv.data + x_grad) - x_ori.data
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)

            x_adv.data = x_ori.data + perturbation
        return x_adv.data
    else:
        raise AttributeError("Provided attack type not supported")
        return 0