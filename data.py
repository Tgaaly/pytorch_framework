import os, glob
import numpy as np

import torch
import torch.utils.data as data

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
		datum.resize((1, datum.shape[0], datum.shape[1]))

		# Currently targets are just the sum of the non-zero pixels.
		targetum = np.zeros((1,), dtype=np.float32)
		targetum[0] = np.mean(datum)

		# Data is by default converted to DoubleTensor. 
		# Here we convert to FloatTensor to save memory.
		datum = torch.FloatTensor(datum)
		targetum = torch.FloatTensor(targetum)
		return datum, targetum
	
	def __len__(self):
		return len(self.file_list)





