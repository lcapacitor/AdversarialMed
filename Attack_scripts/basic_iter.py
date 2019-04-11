import os
import torch
import argparse
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from util import *


CKPT_PATH = 'models/pneu_model.ckpt'
IMG_PATH_NORM = './img/0/'
IMG_PATH_PNEU = './img/1/'
BI_ClASS_NAMES = ['Normal', 'Pneumonia']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def attackBasicIterMethod(data_mode, epsilon, num_iters, alpha):

	# Define loss function
	loss_fn = nn.CrossEntropyLoss().to(device)

	# Load chexnet model
	CheXnet_model = loadPneuModel(CKPT_PATH)

	fileFolder = IMG_PATH_NORM if data_mode==0 else IMG_PATH_PNEU
	files = os.listdir(fileFolder)

	label = data_mode
	label_var = Variable(torch.Tensor([float(label)]).long(), requires_grad=False).to(device)

	for f in files:
		# Get images
		_, img, img_ts = singleImgPreProc(fileFolder + f)

		# Predict with model
		output = CheXnet_model(img_ts)
		_, preds = torch.max(output, 1)
		scores = output.data.cpu().numpy()
		print ('Actual prediction: {}, probs: {}, GT label: {}'.format(BI_ClASS_NAMES[preds], scores, BI_ClASS_NAMES[label]))

		x_adv = img_ts
		ori_img_ts = torch.from_numpy(img).type(torch.float).unsqueeze(0).to(device)

		# Loop over each iteration
		for i in range(num_iters):

			zero_gradients(x_adv)
			output_iter = CheXnet_model(x_adv)
			loss_iter = loss_fn(output_iter, label_var)
			loss_iter.backward()

			x_grad  = alpha * torch.sign(x_adv.grad.data)

			perturbation = (x_adv.data + x_grad) - ori_img_ts.data
			perturbation = torch.clamp(perturbation, -epsilon, epsilon)

			x_adv.data = ori_img_ts + perturbation

			# Predict with adversarial
			f_ouput = CheXnet_model(x_adv)
			_, f_preds = torch.max(f_ouput, 1)
			f_scores = f_ouput.data.cpu().numpy()
			print ('Iter {}/{}, adv prediction: {}, probs: {}'.format(i+1, num_iters, BI_ClASS_NAMES[f_preds], f_scores))

		# Plot results after iterations
		plotFigures(img_ts, preds, scores, x_adv, f_preds, f_scores, perturbation, epsilon)


def main(args):
	epsilon = args.epsilon
	data_type = args.dtype
	num_iters = args.num_iters
	alpha = args.alpha
	attackBasicIterMethod(data_type, epsilon, num_iters, alpha)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dtype', type=int, default=0, help='The data to test with, 0 (default): Normal data, 1: Pneumonia data')
	parser.add_argument('--epsilon', type=float, default=0.05, help='A float value of epsilon, default is 0.05')
	parser.add_argument('--num_iters', type=int, default=10, help='Number of iterations, default is 10')
	parser.add_argument('--alpha', type=float, default=0.005, help='A float value of alpha, default is 0.005')
	args = parser.parse_args()
	main(args)