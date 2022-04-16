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
        - task: The specific task.
        
    Return:
        - The corresponding neural network
    """
    
    if task in ["MNIST", "FMNIST"]:
        return LeNet5(in_channels=1, output_shape=10, dropout_rate=0.25)
    
    if task == "CIFAR10":
        return ResNet18(in_channels=3, output_shape=10)
    
    if task == "CIFAR100":
        return ResNet18(in_channels=3, output_shape=100)
    
    
class FC_Net(nn.Module):
    """Simple fully connected nueral network.""" 

    def __init__(self, input_shape, hidden_layers, n_class, dropout_rate=0.25):
        super().__init__()
        
        if len(hidden_layers) == 0:
            self.model = nn.Linear(input_shape, n_class)
        else:
            layer_list = [nn.Linear(input_shape, hidden_layers[0]),
                          nn.ReLU(),
                          nn.Dropout(dropout_rate)]
            for in_sz, out_sz in zip(hidden_layers[:-1], hidden_layers[1:]):
                layer_list.append(nn.Linear(in_sz, out_sz))   
                layer_list.append(nn.ReLU())    
                layer_list.append(nn.Dropout(dropout_rate))    
            
            layer_list.append(nn.Linear(hidden_layers[-1], n_class))
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
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), 
                            stride=(2, 2), padding=(3, 3), bias=False)
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
        super().__init__()        
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 6, 
                               kernel_size = 5, stride = 1, padding = 2)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, 
                               kernel_size = 5, stride = 1, padding = 2)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, 
                               kernel_size = 3, stride = 1, padding = 1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, output_shape)
        
        self.activation = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.dropout2d = nn.Dropout2d(dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, return_hl=False):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.avgpool(x)
        x = self.dropout2d(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.avgpool(x)
        x = self.dropout2d(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.avgpool2(x)
        x = x.reshape(x.shape[0], -1)
        x_hl = self.linear1(x)
        x = self.activation(x_hl)
        x = self.dropout(x)
        x = self.linear2(x)
        
        if return_hl:
            return x, x_hl
        else:
            return x
    

