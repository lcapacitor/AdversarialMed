import os
import cv2
import copy
import math
import time
import torch
import joblib
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from vae_conv_patch import VAE
import matplotlib.pyplot as plt
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
		x_adv = imgProc(self.adv_fileList[index], is_target=False)
		x_ori = imgProc(self.ori_fileList[index], is_target=False)
		y_tar = imgProc(self.ori_fileList[index], is_target=True)
		#print (int(self.ori_fileList[index].split('/')[-2]))
		# Assertions
		_adv = '_'.join(self.adv_fileList[index].split('/')[-1].split('_')[1:])
		_ori = self.ori_fileList[index].split('/')[-1]
		assert (_adv == _ori)
		assert x_adv.shape == y_tar.shape
		assert (x_adv - x_ori).abs().sum() > 0
		return (x_adv, x_ori), y_tar


def visualResults(adv_img, rec_img, tar_img, epoch):

	assert adv_img.shape == rec_img.shape
	assert rec_img.shape == tar_img.shape
	adv = adv_img.data.cpu()
	adv = adv.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3,1,1)).detach().numpy()
	adv = np.transpose(adv, (1,2,0))   # C X H X W  ==>   H X W X C
	adv = np.clip(adv, 0, 1)

	rec = rec_img.data.cpu().numpy()
	rec = np.transpose(rec, (1,2,0))
	rec = np.clip(rec, 0, 1)

	tar = tar_img.data.cpu().numpy()
	tar = np.transpose(tar, (1,2,0))
	tar = np.clip(tar, 0, 1)

	figure, ax = plt.subplots(1,3)
	ax[0].imshow(adv)
	ax[0].set_title('input adv_img')
	ax[0].axis('off')
	ax[1].imshow(rec)
	ax[1].set_title('output rec_img')
	ax[1].axis('off')
	ax[2].imshow(tar)
	ax[2].set_title('target clean_img')
	ax[2].axis('off')

	outPath = './trained_records/figures/'
	outName = 'vae_val_e{}.png'.format(epoch)
	if not os.path.isdir(outPath):
		os.mkdir(outPath)
	plt.savefig(os.path.join(outPath, outName))
	print ('save fig to: {}'.format(os.path.join(outPath, outName)))


def train(clean_dir, adv_dir, attack_type):
	'''
	clean_dir:
		(str) path of thresholde root folder of all clean images
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
	hidden_dim = 128
	drop_p = 0.5
	image_size = 224
	patch_size = 32
	patch_stride = image_size // patch_size
	channel_num = 3
	is_res = True
	beta = 2

	# Set up training hyer-params
	lr = 5e-4
	weight_decay = 1e-5
	batch_size = 100
	num_epochs = 50
	visual_interval = 2
	best_loss = math.inf
	loss_record = {'train': {'total_loss': [], 'rec_loss':[], 'kl_loss':[]},
 				   'val':   {'total_loss': [], 'rec_loss':[], 'kl_loss':[]}}

	dataset = {x: ImageDataset(clean_dir, adv_dir, attack_type, x) for x in ['train', 'val']}
	dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val']}
	print('Dataset size: train {}, val {}'.format(dataset_sizes['train'], dataset_sizes['val']))

	dataloaders = {'train': DataLoader(dataset['train'], batch_size=batch_size, shuffle=True,  num_workers=0),
                   'val'  : DataLoader(dataset['val'],   batch_size=batch_size, shuffle=False, num_workers=0)}

    # Initialize VAE model, optimizer and scheduler
	model = VAE(patch_size, channel_num, hidden_dim, z_size, is_res, drop_p).to(device)
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

			for in_tuple, targets in tqdm(dataloaders[phase], desc='{} iterations'.format(phase), leave=False):

				for i in range(2):
					inputs = in_tuple[i]
					inputs  = inputs.to(device)
					targets = targets.to(device)

					kl_loss = 0
					rec_loss = 0
					total_loss = 0
	            	# forward-prop
					with torch.set_grad_enabled(phase == 'train'):

						for j in range(patch_stride):
							for i in range(patch_stride):
								row_s = patch_size * j
								row_e = row_s + patch_size
								col_s = patch_size * i
								col_e = col_s + patch_size

								(mean, logvar), reconstructed = model(inputs[:,:,row_s:row_e,col_s:col_e])
								p_kl_loss = model.kl_divergence_loss(mean, logvar)
								p_rec_loss = model.reconstruction_loss(reconstructed, targets[:,:,row_s:row_e,col_s:col_e])
								p_total_loss = p_kl_loss + beta * p_rec_loss

								kl_loss += p_kl_loss.item()
								rec_loss += p_rec_loss.item()
								total_loss += p_total_loss.item()

			                    # backward + optimize only if in training phase
								if phase == 'train':
									# zero the parameter gradients
									optimizer.zero_grad()
									# backward-prop
									p_total_loss.backward()
									optimizer.step()

				# compute loss for running loss
				kl_loss /= patch_stride**2
				rec_loss /= patch_stride**2
				total_loss /= patch_stride**2
				running_kl_loss += kl_loss * inputs.size(0)
				running_rec_loss += rec_loss * inputs.size(0)
				running_total_loss += total_loss * inputs.size(0)

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

			# Save images
			if (epoch+1) % visual_interval == 0 and epoch > 0 and phase == 'val':
				rndIdx = random.randint(0, inputs.size(0)-1)
				print ('Save reconstructed images, random batch index={}'.format(rndIdx))
				rec = torch.zeros((inputs[rndIdx].shape))

				for j in range(patch_stride):
					for i in range(patch_stride):
						row_s = patch_size * j
						row_e = row_s + patch_size
						col_s = patch_size * i
						col_e = col_s + patch_size
						_, reconstructed = model(inputs[rndIdx,:,row_s:row_e,col_s:col_e].unsqueeze(0))
						rec[:,row_s:row_e,col_s:col_e] = reconstructed
				visualResults(inputs[rndIdx], rec, targets[rndIdx], epoch+1)

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
	weight_fname = 'vae_patch{}_{}_zdim{}_hdim{}_e{}_lr{}.torch'.format(patch_size, attack_type, z_size, hidden_dim, num_epochs, str(lr).split('.')[-1])
	s_path = os.path.join(save_path, weight_fname)
	torch.save(best_model_wts, s_path)
	print ('Best weight save to:', s_path)

	save_path = './trained_records/'
	if not os.path.isdir(save_path):
		os.mkdir(save_path)
	weight_fname = 'vae_patch{}_{}_zdim{}_hdim{}_e{}_lr{}.pkl'.format(patch_size, attack_type, z_size, hidden_dim, num_epochs, str(lr).split('.')[-1])
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