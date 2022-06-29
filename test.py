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
model.load_state_dict(torch.load('model.pt'))
model.to(device)

optimizer = SGD(model.parameters(), lr = 0.01)
lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma = 0.1)
criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,),)])
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', download=True, train=False, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=True)

#track the test loss 
test_loss = 0

class_correct = list(0. for i in range (10))
class_total = list(0. for i in range(10))

model.eval()

for images, labels in testloader:
    #forword pass
    images, labels = images.to(device), labels.to(device)
    output = model(images)

    #calculate the loss 
    loss = criterion (output, labels)

    test_loss += loss.item()*images.size(0)

    #convert output probabilities to predicted class

    _, pred = torch.max(output, 1)

    #compare predictions to the true labes

    correct = np.squeeze(pred.eq(labels.data.view_as(pred))) 
    #calculate test accuracy for each object class

    for i in range(len(labels)):

        label = labels.data[i] 
        class_correct[label] += correct[i].item()

        class_total[label] += 1

#calcaulate and print test loss 
test_loss = test_loss/len(testloader.sampler)

print('Test Loss: {:.6f)\n'.format(test_loss))

for i in range(10):

    if class_total[i] > 0:
        print('Test Accuracy of %5s:  %2d%% (%2d/%2d)' %
              (str(i), 100*class_correct[i]/class_total[i],
               np.sum(class_correct[i]), np.sum(class_total[1])))
        
    else:
        print('Test Accuracy of %5s: N/A(no training examples)' % class_total[i])

print('\nTest Accuracy (Overall):  %2d%% (%2d%2d)' % 
      (100. * np.sum(class_correct) /np.sum(class_total), 
       np.sum(class_correct), np.sum(class_total)))