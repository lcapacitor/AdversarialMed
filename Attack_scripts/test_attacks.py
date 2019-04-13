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


from advertorch.defenses import MedianSmoothing2D
from advertorch.defenses import BitSqueezing
from advertorch.defenses import JPEGFilter

from util import *


PNEU_PATH = 'models/pneu_model.ckpt'
CHEX_PATH = 'models/model_14_class.pth.tar'
BI_ClASS_NAMES = ['Normal', 'Pneumonia']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2


def testPerformance(epsilon, data_path, model_type, defense_type):

	# Initialize Defense
	bits_squeezing = BitSqueezing(bit_depth=5)
	median_filter = MedianSmoothing2D(kernel_size=3)
	jpeg_filter = JPEGFilter(23)

	defense_jpeg = nn.Sequential(
		jpeg_filter,
		bits_squeezing,
		median_filter,
	)

	# Load chexnet model
	if model_type == 'pneu':
		model = loadPneuModel(PNEU_PATH)
	if model_type == 'chex':
		model = loadChexnet14(CHEX_PATH)

	# Define loss function
	loss_fn = nn.CrossEntropyLoss()

	# Setup data loader
	data_transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
		])
	image_dataset = datasets.ImageFolder(data_path, data_transform)
	dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

	running_corrects = 0
	pred_probs = []
	gt_labels = []

	running_corrects_adv = 0
	pred_probs_adv = []

	if defense_type is not None:
		running_corrects_defense = 0
		running_corrects_adv_defense = 0
		pred_probs_defense = []
		pred_probs_adv_defense = []

	for inputs, labels in tqdm(dataloader):
		inputs = Variable(inputs).to(device)
		inputs.requires_grad = True

		outputs = model(inputs)
		_, preds = torch.max(outputs, 1)
		pred_probs += outputs[:, 1].tolist()
		gt_labels += labels.tolist()
		running_corrects += torch.sum(preds.cpu() == labels.data).item()

		loss = loss_fn(outputs, labels.to(device))
		loss.backward()

		# FGSM get adversarial
		x_grad = torch.sign(inputs.grad.data)

		perturbation = epsilon * x_grad
		adv_img = inputs.data + perturbation

		# Predict with adversarial
		f_ouput = model(adv_img)
		_, f_preds = torch.max(f_ouput, 1)
		pred_probs_adv += f_ouput[:, 1].tolist()
		running_corrects_adv += torch.sum(f_preds.cpu() == labels.data).item()

		if defense_type is not None:
			if defense_type.lower() == "jpeg":
				defense_input = defense_jpeg(inputs)
				defense_adv = defense_jpeg(adv_img)
			else:
				raise AttributeError("Provided defense type not supported")

			# Predict with defensed images
			defense_outputs = model(defense_input)
			_, defense_preds = torch.max(defense_outputs, 1)
			pred_probs_defense += defense_outputs[:, 1].tolist()
			running_corrects_defense += torch.sum(defense_preds.cpu() == labels.data).item()

			f_ouput_defense = model(defense_adv)
			_, f_preds_defense = torch.max(f_ouput_defense, 1)
			pred_probs_adv_defense += f_ouput_defense[:, 1].tolist()
			running_corrects_adv_defense += torch.sum(f_preds_defense.cpu() == labels.data).item()


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

	if defense_type is not None:
		auc_defense = roc_auc_score(gt_labels, pred_probs_defense)
		auc_adv_defense = roc_auc_score(gt_labels, pred_probs_adv_defense)

		fpr_defense, tpr_defense, thresholds_defense = roc_curve(gt_labels, pred_probs_defense)
		fpr_adv_defense, tpr_adv_defense, thresholds_adv_defense = roc_curve(gt_labels, pred_probs_adv_defense)

		accuracy_defense = running_corrects_defense / len(image_dataset)
		accuracy_adv_defense = running_corrects_adv_defense / len(image_dataset)

		print('Defense on Clean Examples: Accuracy: {:.4f}, AUC: {:.4f}'.format(accuracy_defense, auc_defense))
		print('Defense on Adversarial Examples: Accuracy: {:.4f}, AUC: {:.4f}'.format(accuracy_adv_defense,
																					  auc_adv_defense))

		plt.plot(fpr_defense, tpr_defense, '-.', label='defense on clean (auc = {:.4f})'.format(auc_defense))
		plt.plot(fpr_adv_defense, tpr_adv_defense, '-.',
				 label='defense on adversarial (auc = {:.4f})'.format(auc_adv_defense))


	plt.xlabel('fpr')
	plt.ylabel('tpr')
	plt.legend()
	plt.show()


def main(args):
	epsilon = args.epsilon
	data_path = args.path
	model_type = args.model
	defense_type = args.defense
	testPerformance(epsilon, data_path, model_type, defense_type)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--epsilon', type=float, default=0.02, help='a float value of epsilon, default is 0.02')
	parser.add_argument('--path', type=str, required=True, help='Path to the test images')
	parser.add_argument('--model', type=str, default='pneu', help='specify which model will be tested: pneu or chex, default is pneu')
	parser.add_argument('--defense', type=str, default=None, help='specify which defense to use: JPEG, default is None')
	args = parser.parse_args()
	main(args)
