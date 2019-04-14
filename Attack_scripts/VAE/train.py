import os
import cv2
import copy
import math
import time
import torch
import joblib
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from vae_conv import VAE
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]


def imgProc(img_path, is_target):
	ori_img = cv2.imread(img_path)
	ori_img = cv2.resize(ori_img, (224, 224))
	img = ori_img.copy().astype(np.float32)

	img /= 255.0
	if not is_target:
		img = (img - mean)/std
	img = img.transpose(2, 0, 1)
	img_ts = Variable(torch.from_numpy(img).type(torch.float)).to(device)

	return img_ts


class ImageDataset(Dataset):
	'''
	clean_dir:
		(str) path of the root folder of all clean images
	adv_dir:
		(str) path to the root folder of all attacked images
	attack_type:
		(str) type of attack name
	transform:
		(transforms) transforms for data
	is_train:
		(bool) if the data for training or test
	'''
	def __init__(self, clean_dir, adv_dir, attack_type, setName):
		self.ori_fileList = []
		self.adv_fileList = []
		ori_folder = os.path.join(clean_dir, setName)
		adv_folder = os.path.join(adv_dir, attack_type, setName)
		for path, subdirs, files in os.walk(ori_folder):
			for f in files:
				self.ori_fileList.append(os.path.join(path, f))
				adv_fname = 'adv_' + f
				self.adv_fileList.append(os.path.join(adv_folder, adv_fname))
		assert len(self.ori_fileList)==len(self.adv_fileList)

	def __len__(self):
		return len(self.ori_fileList)

	def __getitem__(self, index):
		x = imgProc(self.ori_fileList[index], is_target=True)
		y = imgProc(self.adv_fileList[index], is_target=True)
		assert x.shape == y.shape
		return x, y


def train(clean_dir, adv_dir, attack_type):
	'''
	clean_dir:
		(str) path of the root folder of all clean images
	adv_dir:
		(str) path to the root folder of all attacked images
	attack_type:
		(str) type of attack name
	'''
	# Ignore all warnings
	import warnings
	warnings.filterwarnings("ignore")

	# Setup Model hyer-param
	z_size = 2048
	hidden_dim = 32
	drop_p = 0.5
	image_size = 224
	channel_num = 3
	is_res = True

	# Set up training hyer-params
	lr = 1e-3
	weight_decay = 1e-5
	batch_size = 64
	num_epochs = 50
	best_loss = math.inf
	loss_record = {'train': {'total_loss': [], 'rec_loss':[], 'kl_loss':[]},
 				   'val':   {'total_loss': [], 'rec_loss':[], 'kl_loss':[]}}

	dataset = {x: ImageDataset(clean_dir, adv_dir, attack_type, x) for x in ['train', 'val']}
	dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val']}
	print('Dataset size: train {}, val {}'.format(dataset_sizes['train'], dataset_sizes['val']))

	dataloaders = {'train': DataLoader(dataset['train'], batch_size=batch_size, shuffle=True,  num_workers=0),
                   'val'  : DataLoader(dataset['val'],   batch_size=batch_size, shuffle=False, num_workers=0)}

    # Initialize VAE model, optimizer and scheduler
	model = VAE(image_size, channel_num, hidden_dim, z_size, is_res, drop_p).to(device)
	optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, threshold=1e-7)

    # Training
	print ('Start training on {}...'.format(device))
	since = time.time()

	for epoch in range(num_epochs):
		print('\nEpoch {}/{}, lr: {}, wd: {}'.format(epoch + 1, num_epochs,
			  optimizer.param_groups[0]['lr'], weight_decay))
		print('-' * 30)

		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()
			else:
				model.eval()

            # Initial running loss
			running_total_loss = 0.0
			running_rec_loss = 0.0
			running_kl_loss = 0.0

			for inputs, targets in tqdm(dataloaders[phase], desc='{} iterations'.format(phase), leave=False):
				inputs  = inputs.to(device)
				targets = targets.to(device)
            	# forward-prop
				with torch.set_grad_enabled(phase == 'train'):
					(mean, logvar), reconstructed = model(inputs)
					rec_loss = model.reconstruction_loss(reconstructed, inputs)
					kl_loss = model.kl_divergence_loss(mean, logvar)
					total_loss = rec_loss + kl_loss

                    # backward + optimize only if in training phase
					if phase == 'train':
						# zero the parameter gradients
						optimizer.zero_grad()
						# backward-prop
						total_loss.backward()
						optimizer.step()

				# compute loss for running loss
				running_kl_loss += kl_loss.item() * inputs.size(0)
				running_rec_loss += rec_loss.item() * inputs.size(0)
				running_total_loss += total_loss.item() * inputs.size(0)

			# Compute epoch loss
			epoch_kl_loss = running_kl_loss / dataset_sizes[phase]
			epoch_rec_loss = running_rec_loss / dataset_sizes[phase]
			epoch_total_loss = running_total_loss / dataset_sizes[phase]

			# Update loss records
			loss_record[phase]['total_loss'].append(epoch_total_loss)
			loss_record[phase]['rec_loss'].append(epoch_rec_loss)
			loss_record[phase]['kl_loss'].append(epoch_kl_loss)

			# Output training/val results
			print('{} Loss: total: {:.4f}, rec_loss: {:.4f}, kl_loss: {:.4f}'
				.format(phase, epoch_total_loss, epoch_rec_loss, epoch_kl_loss))

			# Step optimizer scheduler
			if phase == 'val':
				scheduler.step(epoch_total_loss)

			# Copy best model
			if phase == 'val' and epoch_total_loss < best_loss:
				best_loss = epoch_total_loss
				best_model_wts = copy.deepcopy(model.state_dict())

	# End of training, return the best model
	time_elapsed = time.time() - since
	print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val loss: {}'.format(best_loss))

	# Save the best weights and loss_records
	save_path = './trained_weights/'
	if not os.path.isdir(save_path):
		os.mkdir(save_path)
	weight_fname = 'vae_{}_zdim{}_hdim{}_e{}_lr{}.torch'.format(attack_type, z_size, hidden_dim, num_epochs, str(lr).split('.')[-1])
	s_path = os.path.join(save_path, weight_fname)
	torch.save(best_model_wts, s_path)
	print ('Best weight save to:', s_path)

	save_path = './trained_records/'
	if not os.path.isdir(save_path):
		os.mkdir(save_path)
	weight_fname = 'vae_{}_zdim{}_hdim{}_e{}_lr{}.pkl'.format(attack_type, z_size, hidden_dim, num_epochs, str(lr).split('.')[-1])
	s_path = os.path.join(save_path, weight_fname)
	torch.save(best_model_wts, s_path)
	print ('Training records save to:', s_path)


def main(args):
	clean_dir = args.clean_dir
	adv_dir = args.adv_dir
	attack_type = args.attack_type
	train(clean_dir, adv_dir, attack_type)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--clean_dir', type=str, required=True, help='root folder of the clean images')
	parser.add_argument('--adv_dir', type=str, required=True, help='root folder of the adversarial images')
	parser.add_argument('--attack_type', type=str, required=True, help='type of attacks, e.g. fgsm, b_iter')
	args = parser.parse_args()
	main(args)