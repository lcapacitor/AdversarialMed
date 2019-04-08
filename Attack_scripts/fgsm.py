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
from util import *


CKPT_PATH = 'pneu_model/pneu_model.ckpt'
IMG_PATH_NORM = './img/normal/'
IMG_PATH_PNEU = './img/pneumonia/'
BI_ClASS_NAMES = ['Normal', 'Pneumonia']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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




if __name__ == '__main__':
	attackWithFGSM(epsilon=0.02)

 