"""
5 layer convolutional neural networks on CIFAR-10:
  - 4 convolutional layers (filter size 3 × 3, stride 2)
  - 1 fully-connected layer.
  - Number of feature maps (i.e., channels) are 16, 32, 64 and 128 respectively.

"""

import torch
import torchvision
import torchvision.transforms as transforms
import math
from torch.autograd import Variable
import torch.optim as optim
import matplotlib as plt
import matplotlib.pyplot as plt
import torch.nn as nn
import json
from matplotlib import interactive
from torch.utils.data import sampler

""" Set Number of Epochs """
EPOCH_NUM = 4000
NUM_TRAIN = 2000
NUM_TEST = 500


class ChunkSampler(sampler.Sampler):
   """
   Samples elements sequentially from some offset.
   Arguments:
       num_samples: # of desired datapoints
       start: offset where we should start selecting from
   """
   def __init__(self, num_samples, start = 0):
       self.num_samples = num_samples
       self.start = start

   def __iter__(self):
       return iter(range(self.start, self.start + self.num_samples))

   def __len__(self):
       return self.num_samples

""" Load CIFAR 10 Dataset and set batch size """
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                          sampler=ChunkSampler(NUM_TRAIN,0), num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                         sampler=ChunkSampler(NUM_TEST,0), num_workers=8)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



""" Define network architecture """
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        """ Convolutional layers 1-4. Channels 16,32,64,128. Filter size 3x3, stride 2. """

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()

        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()

        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        """ Fully connected layer """

        self.fc_layer = nn.Sequential(
            nn.Linear(128,10)
        )


    def forward(self, x):
        """Perform forward."""
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc_layer(out)
        return out


net = Net()


""" Error, loss criterion and optimizer - SGD """
def error_criterion(outputs,labels):
    max_vals, max_indices = torch.max(outputs,1)
    error = (max_indices != labels).float().sum()/max_indices.size()[0]
    return error
  
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

""" Check if there is GPU available """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


""" Check if there are multiple GPUs available and parallelize """
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  net = nn.DataParallel(net)

net.to(device)


c1norms = []
c2norms = []
c3norms = []
c4norms = []
f1norms = []

training_loss = []
training_error = []
all_train_loss = []
all_train_error = []
testing_loss = []
testing_error = []
all_test_loss = []
all_test_error = []


c1gammas = []
c2gammas = []
c3gammas = []
c4gammas = []
f1gammas = [] 

running_train_error = 0.0
running_loss = 0.0
test_running_loss = 0.0
test_running_error = 0.0


def train():
    """ Trains function obtains the L2 norms and norm gammas of the weights at every layer"""
    global running_loss
    global running_train_error
    
    c1gammas.append(torch.sum(net.conv_layer1[1].weight))
    
    c2gammas.append(torch.sum(net.conv_layer2[1].weight))
    c3gammas.append(torch.sum(net.conv_layer3[1].weight))
    c4gammas.append(torch.sum(net.conv_layer4[1].weight))
    
    c1norms.append(Variable(torch.sum(torch.norm(net.conv_layer1[0].weight, p=2, dim=1))).item())
    c2norms.append(Variable(torch.sum(torch.norm(net.conv_layer2[0].weight, p=2, dim=1))).item())
    c3norms.append(Variable(torch.sum(torch.norm(net.conv_layer3[0].weight, p=2, dim=1))).item())
    c4norms.append(Variable(torch.sum(torch.norm(net.conv_layer4[0].weight, p=2, dim=1))).item())
    f1norms.append(Variable(torch.sum(torch.norm(net.fc_layer[0].weight, p=2, dim=1))).item())

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_train_error += error_criterion(outputs,labels)

        all_train_loss.append(loss.item())
        all_train_error.append(error_criterion(outputs,labels))

    print('%d loss: %.3f' %(epoch + 1, running_loss))

    training_loss.append(running_loss)
    training_error.append(running_train_error)
    running_loss = 0.0
    running_train_error = 0.0



def test():
  """ Tests function on and records loss and error at every epoch """
  global test_running_loss
  global test_running_error
  with torch.no_grad():
      for data, labels in testloader:
          data, labels = data.to(device), labels.to(device)
          outputs = net(data)
          loss = criterion(outputs, labels)
          test_running_loss += loss.item()
          all_test_loss.append(loss.item())
          test_running_error += error_criterion(outputs,labels)
          all_test_error.append(error_criterion(outputs,labels))
  testing_loss.append(test_running_loss/len(testloader))
  testing_error.append(test_running_error/len(testloader))
  test_running_loss = 0.0
  test_running_error = 0.0


""" Train and test for the given number of Epochs """
for epoch in range(EPOCH_NUM):
    print('Training Epoch', epoch+1)
    train()
    net.eval()
    print('Testing Epoch', epoch+1)
    test()
    net.train()

""" Save the state of the network after training and testing"""
torch.save(net.state_dict(), 'sgd_nobatch_test.ckpt')



""" Plot L2 norms, norm gammas, training and testing loss and error"""
x = list(range(1,EPOCH_NUM+1))

palette = plt.get_cmap('Set1')


plt.figure(figsize=(4, 6))
plt.get_cmap('Set1')
plt.grid()
plt.margins(0)
plt.plot(x, list(map(math.sqrt, c1norms)), color=palette(1), label="conv_layer 1")
plt.plot(x, list(map(math.sqrt,c2norms)), color='red', label="conv_layer 2")
plt.plot(x, list(map(math.sqrt,c3norms)), color=palette(5), label="conv_layer 3")
plt.plot(x, list(map(math.sqrt,c4norms)), color=palette(3), label="conv_layer 4")
plt.plot(x, list(map(math.sqrt,f1norms)), color=palette(2), label="fc_layer 5")
plt.legend(loc=1)
plt.xlabel("Epochs")
plt.ylabel("L2 norm of weights")
interactive(True)
plt.savefig("sgd_nobatch_norms_test.jpg")


plt.figure(figsize=(4, 6))
plt.grid()
plt.margins(0)
plt.plot(x, training_loss, label="Train")
plt.plot(x, testing_loss, color='red', label="Test")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss on CIFAR-10")
interactive(True)
plt.savefig("sgd_nobatch_loss_test.jpg")

plt.figure(figsize=(4, 6))
plt.grid()
plt.margins(0)
plt.plot(x, training_error, label="Train")
plt.plot(x, testing_error, color='red', label="Test")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Error on CIFAR-10")
interactive(True)
plt.savefig("sgd_nobatch_error_test.jpg")


norm_product = []
for a,b,c,d,e in zip(c1norms,c2norms,c3norms,c4norms,f1norms):
  norm_product.append(a*b*c*d*e)

plt.figure(figsize=(4, 6))
plt.grid()
plt.margins(0)
plt.plot(x, norm_product)
plt.legend(loc=1)
plt.xlabel("Epochs")
plt.ylabel("Product of norms")
interactive(True)
plt.savefig("sgd_batch_productnorms_test_500.jpg")


plt.figure(figsize=(4, 6))
plt.get_cmap('Set1')
plt.grid()
plt.margins(0)
plt.plot(x, c1gammas, color=palette(1), label="conv_layer 1")
plt.plot(x, c2gammas, color='red', label="conv_layer 2")
plt.plot(x, c3gammas, color=palette(5), label="conv_layer 3")
plt.plot(x, c4gammas, color=palette(3), label="conv_layer 4")
plt.legend(loc=1)
plt.xlabel("Epochs")
plt.ylabel("Gamma of norms")
interactive(True)
plt.savefig("sgd_nobatch_gammas_test.jpg")    

""" Save results in json file for further analysis """
history = {"c1":c1norms,"c2":c2norms, "c3":c3norms, "c4":c4norms,"f5":f1norms,"loss_train":training_loss,
              "loss_test":testing_loss, "error_train":str(training_error),"error_test":str(testing_error),
              "all_t_loss":str(all_train_loss), "all_t_error":str(all_train_error), "all_test_l": str(all_test_loss),
              "all_test_e":str(all_test_error)}

with open('sgd_nobatch_test.json', 'w') as outfile:
    json.dump(history, outfile)

