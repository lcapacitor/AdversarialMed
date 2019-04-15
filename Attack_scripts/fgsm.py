import os
import util
import torch
import argparse
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable



CKPT_PATH = 'models/pneu_model.ckpt'
IMG_PATH_NORM = './img/0/'
IMG_PATH_PNEU = './img/1/'
BI_ClASS_NAMES = ['Normal', 'Pneumonia']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def attackWithFGSM(epsilon, fileFolder, is_plot, is_save, save_path):

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

		# Predict with model
		output = CheXnet_model(img_ts)
		_, preds = torch.max(output, 1)
		scores = output.data.cpu().numpy()
		print ('Actual prediction: {}, probs: {}, GT label: {}'.format(BI_ClASS_NAMES[preds], scores, BI_ClASS_NAMES[label]))
		
		# Compute loss and its gradients
		loss = loss_fn(output, label_var)
		loss.backward()

		# FGSM get adversarial
		x_grad = torch.sign(img_ts.grad.cpu().data)
		perturbation = epsilon * x_grad
		adv_img = img_ts.cpu().data + perturbation
		adv_img = adv_img.to(device)

		# Predict with adversarial
		f_ouput = CheXnet_model(adv_img)
		_, f_preds = torch.max(f_ouput, 1)
		f_scores = f_ouput.data.cpu().numpy()
		print ('Adv prediction: {}, probs: {}'.format(BI_ClASS_NAMES[f_preds], f_scores))

		# Plot results
		if is_plot:
			util.plotFigures(img_ts, preds, scores, adv_img, f_preds, f_scores, x_grad, epsilon)

		# Save adv_images
		if is_save:
			util.saveImage(save_path, f, adv_img)


def main(args):
	epsilon = args.epsilon
	fileFolder = args.fpath
	is_plot = args.is_plot
	is_save = args.is_save
	save_path = args.save_path
	attackWithFGSM(epsilon, fileFolder, is_plot, is_save, save_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--fpath', type=str, required=True, help='path to the images to attack, e.g. img/0')
	parser.add_argument('--epsilon', type=float, default=0.02, help='a float value of epsilon, default is 0.02')
	parser.add_argument('--is_plot', type=int, default=1, help='if plot the attack results, 1: True, 0: False')
	parser.add_argument('--is_save', type=int, default=0, help='if save the adv_imgs to the given path, 1: True, 0: False')
	parser.add_argument('--save_path', type=str, help='path of where the adv_imgs being saved')
	args = parser.parse_args()
	main(args)