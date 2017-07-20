from torch.utils.data import DataLoader
import sys
import pdb
sys.path.append("../")
from data import FileListDataset

data_dir = '/Users/tarek/Data/KITTI_training_notransform'

if __name__ == '__main__':
	file_list_dataset = FileListDataset(data_dir)
	print len(file_list_dataset)

	dataloader = DataLoader(file_list_dataset, batch_size=4)

	for batch in enumerate(dataloader):
		# returns tuple of batch index and tensor of batchsize x data_size
		print type(batch[1][0])
		print batch[1].size()
		batch_index = batch[0]
		batch_data = batch[1]
		print 'batch_index: ', batch_index
		print 'batch_data: ', batch_data.size

