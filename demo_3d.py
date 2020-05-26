from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from deform_conv_3d import DeformConv3D, DeformConv3D_alternative
import numpy as np

from time import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=5, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

# In this example, the deformable convolution is the first layer - which clearly examples the spatial sampling locations
# Directly on the input image. However, this 3D example is a bit degenerate and does not necessarily allow the
# 3D deformable convolution to truely utilize its 3D capabilities
class DeformNet(nn.Module):
    def __init__(self):
        super(DeformNet, self).__init__()
        self.offsets = nn.Conv3d(1, 81, kernel_size=3, padding=1) # Sampling offsets
        self.conv1 = DeformConv3D(1, 32, kernel_size=3, padding=1) # Uses PyTorch's grid_sample
        #self.conv1 = DeformConv3D_alternative(1, 32, kernel_size=3, padding=1) # Alternative realization
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        # Channels out: 3 * kernel_size**3
        #self.offsets = nn.Conv3d(128, 81, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        d = x.size(4)

        # deformable convolution
        offsets = self.offsets(x)

        # convs
        x = F.relu(self.conv1(x, offsets))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)

        x = F.relu(self.conv4(x))
        x = self.bn4(x)

        x = F.avg_pool3d(x, kernel_size=(28, 28, d), stride=1).view(x.size(0), -1)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1), offsets

class DeformNet2(nn.Module):
    def __init__(self):
        super(DeformNet2, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        # Channels out: 3 * kernel_size**3
        self.offsets = nn.Conv3d(128, 81, kernel_size=3, padding=1)
        self.conv4 = DeformConv3D(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        d = x.size(4)

        # convs
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        # deformable convolution
        offsets = self.offsets(x)
        x = F.relu(self.conv4(x, offsets))
        x = self.bn4(x)

        x = F.avg_pool3d(x, kernel_size=(28, 28, d), stride=1).view(x.size(0), -1)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1), offsets


class PlainNet(nn.Module):
    def __init__(self):
        super(PlainNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        self.conv4 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        d = x.size(4)

        # convs
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)

        x = F.avg_pool3d(x, kernel_size=(28, 28, d), stride=1).view(x.size(0), -1)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)

model = DeformNet()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            m.bias.data = torch.FloatTensor(m.bias.shape[0]).zero_()


def init_conv_offset(m):
    #m.weight.data = torch.zeros_like(m.weight.data)
    nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu')) # OS: test
    if m.bias is not None:
        m.bias.data = torch.FloatTensor(m.bias.shape[0]).zero_()


model.apply(init_weights)
model.offsets.apply(init_conv_offset)

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def augment_3d(input, nz=3, axis=4):
    internal_augmentation_flag = 'circ_repeat'
    if internal_augmentation_flag == 'repeat':
        # Copy the image nz times along the depth dimension
        tmp = input.unsqueeze(dim=-1)
        output = tmp.repeat_interleave(repeats=nz, dim=axis)
    elif internal_augmentation_flag == 'circ_repeat':
        # Create a rotated copy with increasing rotation angle of the image and append to depth dimension
        from skimage.transform import rotate
        b, c, h, w = input.size()
        tmp_out = np.zeros((b, c, h, w, nz))
        b_size = input.size(0)
        rot_degrees = [180./f for f in reversed(range(1, nz))] # Divide to rotation angles with fixed increment [deg]
        rot_degrees.insert(0, 0.0)

        # Apply rotations
        for bb in range(b_size):
            for zz in range(nz):
                array = input[bb, 0, :, :]
                #img = Image.fromarray(np.uint8(cm.gist_earth(array)*255))
                #img = tvF.rotate(img=img, angle=rot_degrees[zz])
                img = rotate(image=array, angle=rot_degrees[zz])
                img[img < 0] = 0
                tmp_out[bb, 0, :, :, zz] = img
        output = torch.tensor(tmp_out).type(torch.FloatTensor)

    return output

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Augment data (only) to 3D cube
        data = augment_3d(data)

        data, target = Variable(data), Variable(target)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output, offsets = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # Augment data (only) to 3D cube
        data = augment_3d(data)

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output, offsets = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # Test visually
        plot_3d(data, offsets, x=14, y=14, batch_ind=1)
        #

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def plot_3d(data, offsets, x=14, y=14, batch_ind=1):
    '''
    Plots the sampling points for pixel (x, y) over all depth values z over the input volume
    '''
    import matplotlib.pyplot as plt
    b_size = data.size(-1)
    N = int(offsets.size(1) / 3)

    offs_list = []
    data_list = []
    for zz in range(offsets.size(-1)):
        offs_list.append(offsets[batch_ind, :, x, y, zz].detach().cpu().numpy())
        data_list.append(data[batch_ind, 0, :, :, zz].detach().cpu().numpy())

    fig, axes = plt.subplots(nrows=2, ncols=3)
    fig.subplots_adjust(hspace=1)
    fig.suptitle('x = {}, y = {}'.format(x, y))

    z = 0
    for ax, img, offs in zip(axes.flatten(), data_list, offs_list):
        ax.imshow(img, cmap='gray')
        #color = [str(item / 255.) for item in (z + offs[2*N:])]
        #ax.scatter(x + offs[0:N], y + offs[N:2*N], c=color)
        im = ax.scatter(x + offs[0:N], y + offs[N:2 * N], c=(offs[2*N:]) / 255., cmap='viridis', s=30)
        ax.set(title='z = {}'.format(z))
        #if z == b_size:
        fig.colorbar(im, ax=ax, orientation='vertical')
        z += 1
    plt.show()

for epoch in range(1, args.epochs + 1):
    print('Epoch = {}'.format(epoch))
    since = time()
    train(epoch)
    iter = time() - since
    print("Spends {}s for each training epoch".format(iter/args.epochs))
    test()
