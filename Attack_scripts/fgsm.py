import os
import torch
import argparse
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


def attackWithFGSM(epsilon, data_mode):

	# Define loss function
	loss_fn = nn.CrossEntropyLoss().to(device)

	# Load chexnet model
	CheXnet_model = loadChexNetModel(CKPT_PATH)

	fileFolder = IMG_PATH_NORM if data_mode==0 else IMG_PATH_PNEU
	files = os.listdir(fileFolder)

	label = data_mode
	label_var = Variable(torch.Tensor([float(label)]).long(), requires_grad=False).to(device)

	for f in files:
		# Get images
		_, _, img_ts = singleImgPreProc(fileFolder + f)

		# Predict with model
		output = CheXnet_model(img_ts)
		_, preds = torch.max(output, 1)
		scores = output.data.cpu().numpy()
		print ('Actual prediction: {}, probs: {}, GT label: {}'.format(BI_ClASS_NAMES[preds], scores, BI_ClASS_NAMES[label]))
		
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
		print ('Adv prediction: {}, probs: {}'.format(BI_ClASS_NAMES[f_preds], f_scores))

		# Plot results
		plotFigures(img_ts, preds, scores, adv_img, f_preds, f_scores, x_grad, epsilon)


def main(args):
	epsilon = args.epsilon
	data_type = args.dtype
	attackWithFGSM(epsilon, data_type)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dtype', type=int, default=0, help='type of data to test with, 0 (default): Normal data, 1: Pneumonia data')
	parser.add_argument('--epsilon', type=float, default=0.02, help='a float value of epsilon, default is 0.02')
	args = parser.parse_args()
	main(args)

	

 