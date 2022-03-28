import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class FC_Net(nn.Module):
    """Simple fully connected nueral network.""" 

    def __init__(self, input_shape, output_shape, dropout_rate=0.25):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, 128)  # 5*5 from image dimension
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_shape)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        #x = F.softmax(self.fc3(x), dim=1)
        return x
    
def ResNet18(in_channels, n_class, pretrained=False):
    """Create a personalized model base on the ResNet18 model archtecture.
    
    Arguments:
        -in_channels: Number of input channels
        -n_class: Number of class (for the output dimention)
    
    Return:
        -model: A Pytorch ResNet18 model
    """
    # Load ResNet18
    model = torchvision.models.resnet18(pretrained=pretrained)
    # Adapt input layer
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), 
                            stride=(2, 2), padding=(3, 3), bias=False)
    # Adapt output layer
    model.fc = nn.Linear(in_features=512, out_features=n_class, bias=True)
    
    return model