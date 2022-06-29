import torch
import numpy as np
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model.resnet import ResNet50

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Data download and preprocessing

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,),)])

trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', download=True, train=True, transform=transform)
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', download=True, train=False, transform=transform)

indices = list(range(len(trainset)))
np.random.shuffle(indices)
split = int(np.floor(0.2 * len(trainset)))

train_sample = SubsetRandomSampler(indices[:split])
valid_sample = SubsetRandomSampler(indices[split:])

trainloader = DataLoader(trainset, sampler = train_sample, batch_size=64)
validloader = DataLoader(trainset, sampler=valid_sample, batch_size=64)
testloader = DataLoader(testset, batch_size=64, shuffle=True)

# Defining model parameter and losses

num_classes = 10

model = ResNet50(image_depth = 1, num_classes = num_classes)

model.linear = nn.Sequential(
    nn.Linear(
        in_features = 2048,
        out_features = num_classes
    ),
    nn.Sigmoid()
)
model.to(device)

optimizer = SGD(model.parameters(), lr = 0.01)
lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma = 0.1)
criterion = nn.CrossEntropyLoss()

# training

valid_loss_min = np.Inf
epochs = 20
steps = 0
model.train()
train_losses, valid_losses = [], []

for e in range(epochs):
    running_loss = 0
    valid_loss = 0
    
    for images, labels in trainloader:
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*images.size(0)
    
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        valid_loss += loss.item()*images.size(0)
    
    running_loss = running_loss/len(trainloader.sampler)
    valid_loss = valid_loss/len(validloader.sampler)
    train_losses.append(running_loss)
    valid_losses.append(valid_loss)
    
    print('Epoch: {}\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(e+1, running_loss, valid_loss))
    
    if valid_loss <= valid_loss_min:
        print('validation loss decreased({:.6f} --> {:.6f}). Saving Model ...'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = valid_loss  
        
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Comparisons")
plt.show()