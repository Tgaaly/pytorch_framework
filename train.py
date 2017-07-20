from data import FileListDataset
from torch.utils.data import DataLoader
import sys
import pdb
import torch.optim as optim
from model import Net
from torch.autograd import Variable
import torch
import numpy as np
import torch.nn.functional as F

# Data loading.
data_dir = '/Users/tarek/Data/KITTI_training_notransform'
file_list_dataset = FileListDataset(data_dir)
train_loader = DataLoader(file_list_dataset, batch_size=4, num_workers=4)


num_epochs=10
log_interval=5

# set to False for working on macbook.
args.cuda=False

# Instantiate the model (transfer to GPU is cuda support is enabled).
model = Net()
if args.cuda:
     model.cuda()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.09)

# Training function.
def train(epoch):
    model.train()
    for (batch_idx, batch_data) in enumerate(train_loader):
        
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
        # Create pytorch Variable for data batch.
        data = Variable(batch_data)
 
 		# Create pytorch Variable for target batch.
        batch_target = np.random.randint(11, size=(4,1))
        target = Variable(torch.FloatTensor(batch_target))

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
        if batch_idx % log_interval == 0:
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


for epoch in range(1, num_epochs + 1):
    train(epoch)
    test()

