import os
import cv2
import foolbox
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable


CKPT_PATH = 'pneu_model.ckpt'
CHEXNET_PATH = 'model_14_class.pth.tar'
IMG_PATH_NORM = './img/normal/'
IMG_PATH_PNEU = './img/pneumonia/'
BI_ClASS_NAMES = ['Normal', 'Pneumonia']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]



def loadChexNetModel(model_path):
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


def imgPreProc(img_path):

	ori_img = cv2.imread(img_path)
	ori_img = cv2.resize(ori_img, (224, 224))
	img = ori_img.copy().astype(np.float32)

	img /= 255.0
	img = (img - mean)/std
	img = img.transpose(2, 0, 1)

	img_ts = Variable(torch.from_numpy(img).type(torch.float).unsqueeze(0), requires_grad=True).to(device)

	return ori_img, img, img_ts



def attackWithFGSM(epsilon):

	# Define loss function
	loss_fn = nn.CrossEntropyLoss().to(device)

	# Load chexnet model
	CheXnet_model = loadChexNetModel(CKPT_PATH)

	fileFolder = IMG_PATH_NORM
	files = os.listdir(fileFolder)

	for f in files:
		# Get images
		_, _, img_ts = imgPreProc(fileFolder + f)

		# Predict with model
		output = CheXnet_model(img_ts)
		_, preds = torch.max(output, 1)
		scores = output.data.cpu().numpy()
		print ('Actual prediction:', scores, BI_ClASS_NAMES[preds])
		label_var = Variable(torch.Tensor([float(preds)]).long(), requires_grad=False).to(device)

		# Compute loss and its gradients
		loss = loss_fn(output, label_var)
		loss.backward()

		# FGSM get adversarial
		x_grad = torch.sign(img_ts.grad.data)
		perturbation = epsilon * x_grad
		adv_img = img_ts.data + perturbation

		# Predict with adversarial
		f_ouput = CheXnet_model(adv_img)
		_, f_preds = torch.max(f_ouput, 1)
		f_scores = f_ouput.data.cpu().numpy()
		print ('Adv prediction:', f_scores, BI_ClASS_NAMES[f_preds])

		# Plot results
		plotFigures(img_ts, preds, scores, adv_img, f_preds, f_scores, x_grad, epsilon)



def plotFigures(oriImg, preds, ori_score, advImg, f_preds, f_score, x_grad, epsilon):

	# Get original image
	x = oriImg.squeeze(0)
	x = x.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3,1,1)).detach().numpy()
	x = np.transpose(x, (1,2,0))   # C X H X W  ==>   H X W X C
	x = np.clip(x, 0, 1)

    	# Get adversarail image
	x_adv = advImg.squeeze(0)
	x_adv = x_adv.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3,1,1)).detach().numpy()
	x_adv = np.transpose(x_adv, (1,2,0))   # C X H X W  ==>   H X W X C
	x_adv = np.clip(x_adv, 0, 1)

	# Get perturbation image
	x_grad = x_grad.squeeze(0).numpy()
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



if __name__ == '__main__':
	attackWithFGSM(epsilon=0.02)

 