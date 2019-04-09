import os
import torch
import argparse
import torchvision
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, roc_auc_score

from util import *


CKPT_PATH = 'pneu_model/pneu_model.ckpt'
BI_ClASS_NAMES = ['Normal', 'Pneumonia']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16


def testPerformance(epsilon, data_path):

	# Load chexnet model
	CheXnet_model = loadChexNetModel(CKPT_PATH)

	# Define loss function
	loss_fn = nn.CrossEntropyLoss()

	# Setup data loader
	data_transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
		])
	image_dataset = datasets.ImageFolder(data_path, data_transform)
	dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

	running_corrects = 0
	pred_probs = []
	gt_labels = []

	running_corrects_adv = 0
	pred_probs_adv = []

	for inputs, labels in tqdm(dataloader):
		inputs = Variable(inputs, requires_grad=True)

		outputs = CheXnet_model(inputs)
		_, preds = torch.max(outputs, 1)
		pred_probs += outputs[:, 1].tolist()
		gt_labels += labels.tolist()
		running_corrects += torch.sum(preds == labels.data)

		loss = loss_fn(outputs, labels)
		loss.backward()

		# FGSM get adversarial
		x_grad = torch.sign(inputs.grad.data)

		perturbation = epsilon * x_grad
		adv_img = inputs.data + perturbation

		# Predict with adversarial
		f_ouput = CheXnet_model(adv_img)
		_, f_preds = torch.max(f_ouput, 1)
		pred_probs_adv += f_ouput[:, 1].tolist()
		running_corrects_adv += torch.sum(f_preds == labels.data)

	# compute metrices
	auc = roc_auc_score(gt_labels, pred_probs)
	auc_adv = roc_auc_score(gt_labels, pred_probs_adv)
	fpr, tpr, thresholds = roc_curve(gt_labels, pred_probs)
	fpr_adv, tpr_adv, thresholds_adv = roc_curve(gt_labels, pred_probs_adv)
	accuracy = running_corrects / len(image_dataset)
	accuracy_adv = running_corrects_adv / len(image_dataset)

	print ('Clean Examples: Accuracy: {:.4f}, AUC: {:.4f}'.format(accuracy, auc))
	print ('Adversarial Examples: Accuracy: {:.4f}, AUC: {:.4f}'.format(accuracy_adv, auc_adv))

	plt.plot(fpr, tpr, '-.', label='clean (auc = {:.4f})'.format(auc))
	plt.plot(fpr_adv, tpr_adv, '-.', label='adversarial (auc = {:.4f})'.format(auc_adv))

	plt.xlabel('fpr')
	plt.ylabel('tpr')
	plt.legend()
	plt.show()


def main(args):
	epsilon = args.epsilon
	data_path = args.path
	testPerformance(epsilon, data_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--epsilon', type=float, default=0.02, help='a float value of epsilon, default is 0.02')
	parser.add_argument('--path', type=str, required=True, help='Path to the test images')
	args = parser.parse_args()
	main(args)

