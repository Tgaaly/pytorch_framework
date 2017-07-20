import os
import numpy as np
import torch.utils.data as data
import glob
import torch

# data_dir = '/Users/tarek/Data/KITTI_training_notransform'

def get_file_list(data_dir):
	print 'getting all files: ', os.path.join(data_dir, '*.npy')
	files = glob.glob(os.path.join(data_dir, '*.npy'))
	return files

class FileListDataset(data.Dataset):
	
	def __init__(self, data_dir):
		self.data_dir = data_dir
		self.file_list = get_file_list(data_dir)
		print len(self.file_list), 'files read'
		# self.transform = transform
	
	def __getitem__(self, idx):
		filepath = self.file_list[idx]
		f = os.path.join(self.data_dir,filepath)
		datum = np.load(f)
		datum = datum[1]
		# print datum.shape
		datum.resize((1, datum.shape[0], datum.shape[1]))
		# print datum.shape
		# datum = torch.Tensor(datum)
		# print datum.shape
		datum = torch.FloatTensor(datum)
		return datum
	
	def __len__(self):
		return len(self.file_list)





