import os
import util
import torch
import argparse
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients


CKPT_PATH = 'models/pneu_model_aug.ckpt'
IMG_PATH_NORM = './img/0/'
IMG_PATH_PNEU = './img/1/'
BI_ClASS_NAMES = ['Normal', 'Pneumonia']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def attackBasicIterMethod(fileFolder, epsilon, num_iters, alpha, is_plot, is_save, save_path):

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
		_, img, img_ts = util.singleImgPreProc(os.path.join(fileFolder, f))

		# Predict with model
		output = CheXnet_model(img_ts)
		_, preds = torch.max(output, 1)
		scores = output.data.cpu().numpy()
		print ('Actual prediction: {}, probs: {}, GT label: {}'.format(BI_ClASS_NAMES[preds], scores, BI_ClASS_NAMES[label]))

		'''
		x_adv = util.generateAdvExamples(CheXnet_model, loss_fn, label_var, img_ts, epsilon, num_iters, alpha, attack_type='b_iter')

		'''
		x_adv = Variable(img_ts.data, requires_grad=True)
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

		# Predict with adversarial
		f_ouput = CheXnet_model(x_adv)
		_, f_preds = torch.max(f_ouput, 1)
		f_scores = f_ouput.data.cpu().numpy()
		print ('Adv prediction: {}, probs: {}'.format(BI_ClASS_NAMES[f_preds], f_scores))
		
		# Plot results
		if is_plot:
			util.plotFigures(img_ts, preds, scores, x_adv, f_preds, f_scores, perturbation, epsilon)

		# Save adv_images
		if is_save:
			util.saveImage(save_path, f, x_adv)


def main(args):
	epsilon = args.epsilon
	fileFolder = args.fpath
	num_iters = args.num_iters
	alpha = args.alpha
	is_plot = args.is_plot
	is_save = args.is_save
	save_path = args.save_path
	attackBasicIterMethod(fileFolder, epsilon, num_iters, alpha, is_plot, is_save, save_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--fpath', type=str, required=True, help='path to the images to attack, e.g. img/0')
	parser.add_argument('--epsilon', type=float, default=0.05, help='A float value of epsilon, default is 0.05')
	parser.add_argument('--num_iters', type=int, default=10, help='Number of iterations, default is 10')
	parser.add_argument('--alpha', type=float, default=0.005, help='A float value of alpha, default is 0.005')
	parser.add_argument('--is_plot', type=int, default=1, help='if plot the attack results, 1: True, 0: False')
	parser.add_argument('--is_save', type=int, default=0, help='if save the adv_imgs to the given path, 1: True, 0: False')
	parser.add_argument('--save_path', type=str, help='path of where the adv_imgs being saved')
	args = parser.parse_args()
	main(args)