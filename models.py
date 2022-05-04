# -----------------------------------------------------------------------------
# This file contains the helper code that was developped
# during my master thesis at MIT.
#
# 2022 Frédéric Berdoz, Boston, USA
# -----------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def get_model(task):
    """Build a model specific for the given task.
    
    Arguments:
        - task: The specific task (dataset).
        
    Return:
        - The corresponding neural network
    """
    
    if task in ["MNIST", "FMNIST"]:
        return LeNet5(in_channels=1, output_shape=10, dropout_rate=0.25)
    
    if task == "CIFAR10":
        return ResNet9(in_channels=3, output_shape=10)
    
    if task == "CIFAR100":
        return ResNet9(in_channels=3, output_shape=100)
    
    
class FC_Net(nn.Module):
    """Simple fully connected nueral network.""" 

    def __init__(self, input_shape, hidden_layers, output_shape, dropout_rate=0.25):
        super().__init__()
        
        if len(hidden_layers) == 0:
            self.model = nn.Linear(input_shape, output_shape)
        else:
            layer_list = [nn.Linear(input_shape, hidden_layers[0]),
                          nn.ReLU(),
                          nn.Dropout(dropout_rate)]
            for in_sz, out_sz in zip(hidden_layers[:-1], hidden_layers[1:]):
                layer_list.append(nn.Linear(in_sz, out_sz))   
                layer_list.append(nn.ReLU())    
                layer_list.append(nn.Dropout(dropout_rate))    
            
            layer_list.append(nn.Linear(hidden_layers[-1], output_shape))
            self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.model(x)
        
        return x
    

def ResNet18(in_channels, output_shape, pretrained=False):
    """Create a personalized model base on the ResNet18 model archtecture.
    
    Arguments:
        -in_channels: Number of input channels
        -output_shape: Number of output dimension.
    
    Return:
        -model: A Pytorch ResNet18 model
    """
    # Load ResNet18
    model = torchvision.models.resnet18(pretrained=pretrained)
    # Adapt input layer
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # Adapt output layer
    model.fc = nn.Linear(in_features=512, out_features=output_shape, bias=True)
    
    return model

class LeNet5(nn.Module):
    def __init__(self, in_channels, output_shape, dropout_rate=0.25):
        """Create a personalized model base on the LeNet5 model archtecture.
    
        Arguments:
            -in_channels: Number of input channels
            -output_shape: Number of class (for the output dimention)
            -dropout_rate: Percentage of neurons to drop.
        """
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels = in_channels, out_channels=6, kernel_size=5, stride=1, padding=2),
                                      nn.ReLU(),
                                      nn.AvgPool2d(kernel_size=2, stride=2),
                                      nn.Dropout2d(dropout_rate),
                                      nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2),
                                      nn.ReLU(),
                                      nn.AvgPool2d(kernel_size=2, stride=2),
                                      nn.Dropout2d(dropout_rate),
                                      nn.Conv2d(in_channels=16, out_channels=120, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                      nn.Flatten(start_dim=1),
                                      nn.Linear(120, 84),
                                      nn.Tanh())
        self.classifier = nn.Linear(84, output_shape)


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    


class ResBlock(nn.Module):
    """
    A residual block (He et al. 2016)
    Inspired by https://github.com/matthias-wright/cifar10-resnet/blob/master/model.py
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        if stride != 1:
            # Downsample residual in case stride > 1
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            res = self.downsample(res)

        x = self.relu(x)
        x = x + res
        return x


class ResNet9(nn.Module):
    """
    Residual network with 9 layers.
    """
    def __init__(self, in_channels, output_shape):
        super(ResNet9, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1))
                                      
        self.classifier = nn.Linear(in_features=256, out_features=output_shape, bias=True)
        

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x