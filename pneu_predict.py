import os
import re
import copy
import time
import foolbox
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imread
from skimage.transform import resize

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable

from pneu_train import trained_chexnet
from AttackFGS import AttackFGS
from Attacks import FGSMAttack, LinfPGDAttack


CKPT_PATH = 'pneu_model.ckpt'
CHEXNET_PATH = 'model_14_class.pth.tar'
IMG_PATH_NORM = './img/normal/'
IMG_PATH_PNEU = './img/pneumonia/'
ORI_CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', \
					'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
BI_ClASS_NAMES = ['Normal', 'Pneumonia']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def loadChexNetModel(path):
	out_size = 2
	model = torchvision.models.densenet121(pretrained=True)
	num_ftrs = model.classifier.in_features
	model.classifier = nn.Sequential(
		nn.Linear(num_ftrs, out_size),
		nn.Softmax(1)
	)

	# load checkpoint
	if torch.cuda.is_available():
	    state_dict = torch.load(path)
	else:
	    state_dict = torch.load(path, map_location='cpu')

	model.load_state_dict(state_dict)
	return model.to(device)


def imgProc(imgPath):
	# Load image
	img = Image.open(imgPath).convert('RGB')
	# Transform
	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		#transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		])
	img_ts = transform(img)

	# To tensor
	img_ts = img_ts.view(1, 3, 224, 224)
	input_var = Variable(img_ts, requires_grad=True)
	return input_var.to(device)


def predictChex():
	# Load chexnet model
	CheXnet_model = loadChexNetModel(CKPT_PATH)
	#CheXnet_model = trained_chexnet(CHEXNET_PATH)
	CheXnet_model.eval()

	# Create fmodel
	fmodel = foolbox.models.PyTorchModel(CheXnet_model, bounds=(0, 1), num_classes=2)
	attack = foolbox.attacks.FGSM(fmodel)

	fgs = AttackFGS(targeted=False, optimize_epsilon=False, max_epsilon=0.3, num_iter=20)
	fgsm = FGSMAttack(CheXnet_model, epsilon=0.3)

	mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
	std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

	# Do tests
	fileFolder = IMG_PATH_PNEU
	files = os.listdir(fileFolder)
	for f in files:
		# Get input
		input = imgProc(fileFolder + f)
		image_arr = input.squeeze(0).data.cpu().numpy()

		# Get label
		if fileFolder == IMG_PATH_NORM:
			label = 0
		else:
			label = 1
		label_var = Variable(torch.tensor([label]).type(torch.long)).to(device)

		# Predict with model
		output = CheXnet_model(input)
		_, preds = torch.max(output, 1)
		scores = output.data.cpu().numpy()
		print ('Actual predict', scores, BI_ClASS_NAMES[preds])

		# Get adversarial
		adversarial = attack(image_arr, label_var)
		#adversarial = fgs.generate_ad_ex(CheXnet_model, input, label_var)
		#adversarial = fgsm.perturb(input, label_var)

		# Use CheXnet_model to predict adversarial examples  
		#adv_tensor = torch.from_numpy(adversarial).view(1, 3, 224, 224)
		f_ouput = CheXnet_model(adversarial).data.cpu().numpy()
		f_predict = np.argmax(f_ouput)

		'''
		# Use FoolBox to predict
		f_ouput = fmodel.predictions(adversarial)
		f_predict = np.argmax(f_ouput)
		'''
		print ('fmodel predict', f_ouput, BI_ClASS_NAMES[f_predict])

		# Plot
		plotFigures(input, scores, adversarial, f_ouput)
		


def plotFigures(oriImg, ori_score, advImg, f_score):

	advImg_np = advImg.squeeze(0).data.cpu().numpy()
	oriImg = oriImg.squeeze(0).data.cpu().numpy()

	# Move the channel dimension from the front to the end
	oriImg = np.moveaxis(oriImg, 0, -1)
	advImg = np.moveaxis(advImg_np, 0, -1)

	oriImg_1d = oriImg[1]
	advImg_1d = advImg[1]
	plt.figure()
	# Plot the original image
	plt.subplot(1, 3, 1)
	plt.title('Original\n{:.2f}% {}'.format(np.max(ori_score) * 100, BI_ClASS_NAMES[np.argmax(ori_score)]))
	#plt.imshow(oriImg_1d, cmap='gray')
	plt.imshow(oriImg)
	plt.axis('off')

	# Plot the adversarial image
	plt.subplot(1, 3, 2)
	plt.title('Adversarial\n{:.2f}% {}'.format(np.max(f_score) * 100, BI_ClASS_NAMES[np.argmax(f_score)]))
	#plt.imshow(advImg_1d, cmap='gray')
	plt.imshow(advImg)
	plt.axis('off')

	# Plos the difference
	diff = oriImg - advImg
	plt.subplot(1, 3, 3)
	plt.title('Adversarial examples')
	plt.imshow(diff)
	plt.axis('off')


	plt.show()
	

def main():
	predictChex()


if __name__ == '__main__':
	main()
