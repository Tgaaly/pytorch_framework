from data import FileListDataset
from torch.utils.data import DataLoader
import sys
import pdb

data_dir = '/Users/tarek/Data/KITTI_training_notransform'

# data_loader = FileLoader(data_dir)
file_list_dataset = FileListDataset(data_dir)
print len(file_list_dataset)
# pdb.set_trace()
# train_loader = DataLoader(FileLoader(data_dir))

# for i in range(len(file_list_dataset)):
# 	file_numpy = file_list_dataset[i]
# 	# print i, file_numpy.shape

dataloader = DataLoader(file_list_dataset, batch_size=4)

for batch in enumerate(dataloader):
	# returns tuple of batch index and tensor of batchsize x data_size
	print type(batch[1][0])
	print batch[1].size()
	batch_index = batch[0]
	batch_data = batch[1]
	print 'batch_index: ', batch_index
	# pdb.set_trace()

# for i, datum in enumerate(train_loader):
# 	print i, dataum.shape
