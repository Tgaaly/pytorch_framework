import sys
import numpy as np
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import Net
from data import FileListDataset

# Data loading.
data_dir = '/Users/tarek/Data/KITTI_training_notransform'
file_list_dataset = FileListDataset(data_dir)
train_loader = DataLoader(file_list_dataset, batch_size=4, num_workers=4)

# Training settings.
parser = argparse.ArgumentParser(description='PyTorch Simple Framework')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--num_epochs', type=int, default=10, metavar='N',
                    help='Number of epochs to train')
parser.add_argument('--log_interval', action='store_true', default=5,
                    help='How often to log training information')

# set to False for working on macbook.
args = parser.parse_args()

# Instantiate the model (transfer to GPU is cuda support is enabled).
model = Net()
if args.cuda:
     model.cuda()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.09)

# Training function.
def train(epoch):
    model.train()
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
        # Create pytorch Variable for data and target batch.
        data = Variable(batch_data)
        target = Variable(batch_target)
 
        # Zero-fy the gradients.
        optimizer.zero_grad()

        # Feed-forward and get output of net.
        output = model(data)

        # Compute loss.
        loss_fn = torch.nn.MSELoss(size_average=True)
        loss = loss_fn(output, target)

        # Backprop.
        loss.backward()

        # Update weights using the gradients computed in backprop.
        optimizer.step()

        # Log training performance.
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()
    # test_loss = 0
    # correct = 0
    # for data, target in test_loader:
    #     if args.cuda:
    #         data, target = data.cuda(), target.cuda()
    #     data, target = Variable(data, volatile=True), Variable(target)
    #     output = model(data)
    #     test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
    #     pred = output.data.max(1)[1] # get the index of the max log-probability
    #     correct += pred.eq(target.data).cpu().sum()

    # test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    for epoch in range(1, args.num_epochs + 1):
        train(epoch)
        test()

