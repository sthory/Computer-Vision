## We define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional Layers battery
        
        self.conv1 = nn.Conv2d(in_channels = 1, 
                               out_channels = 32, 
                               kernel_size =4)
        
        self.conv2 = nn.Conv2d(in_channels = 32, 
                               out_channels = 64, 
                               kernel_size =3)
        
        self.conv3 = nn.Conv2d(in_channels = 64, 
                               out_channels = 128, 
                               kernel_size =2)
        
        self.conv4 = nn.Conv2d(in_channels = 128, 
                               out_channels = 256, 
                               kernel_size =1)
        
        # Maxpooling Layer battery
        
        self.pool1 = nn.MaxPool2d(kernel_size = 2, 
                                  stride = 2)

        self.pool2 = nn.MaxPool2d(kernel_size = 2, 
                                  stride = 2)
        
        self.pool3 = nn.MaxPool2d(kernel_size = 2, 
                                  stride = 2)
        
        self.pool4 = nn.MaxPool2d(kernel_size = 2, 
                                  stride = 2)
        
        # Fully Connected Layers (Dense) Battery
        
        
        # output size = (W-F)/S +1 = (26-1)/1 +1 = 26
        # output size after pooling = (26 - 2)/2 + 1 = 13 <= see calculation below (the why)
        self.fc1 = nn.Linear(in_features = 256*13*13, # <= see calculation below (the why)
                             out_features = 1000)
        
        self.fc2 = nn.Linear(in_features = 1000,
                             out_features = 1000)
                             
        self.fc3 = nn.Linear(in_features = 1000,
                             out_features = 136) # the output 136  
                                                 # according to having 2
                                                 # for each of the 68
                                                 # keypoint (x, y) pairs
        
        # Dropouts Battery (according NaimishNet)
        
        self.drop1 = nn.Dropout(p = 0.1)
        
        self.drop2 = nn.Dropout(p = 0.2)
        
        self.drop3 = nn.Dropout(p = 0.3)
        
        self.drop4 = nn.Dropout(p = 0.4)
        
        self.drop5 = nn.Dropout(p = 0.5)
        
        self.drop6 = nn.Dropout(p = 0.6)
 
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # W = image width
        # F = filter size
        # S = stride
        
        # First step: Convolution, activation, pooling and dropout
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))          
        # After first step:
        # output size = (W-F)/S +1 = (224-4)/1 +1 = 221
        # output size after pooling = (221 - 2)/2 + 1 = 110
        # the output Tensor for one image:
        #        will have the dimensions: (32, 110, 110)
        
        # Second step: Convolution, activation, pooling and dropout
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        # After second step:
        # output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # output size after pooling = (108 - 2)/2 + 1 = 54
        # the output Tensor for one image:
        #        will have the dimensions: (64, 54, 54)
        
        # Third step: Convolution, activation, pooling and dropout
        x = self.drop3(self.pool3(F.relu(self.conv3(x))))
        # After third step:
        # output size = (W-F)/S +1 = (54-2)/1 +1 = 53
        # output size after pooling = (53 - 2)/2 + 1 = 26
        # the output Tensor for one image:
        #        will have the dimensions: (128, 26, 26)
        
        # Fourth step: Convolution, activation, pooling and dropout
        x = self.drop4(self.pool4(F.relu(self.conv4(x))))
        # After fourth step:
        # output size = (W-F)/S +1 = (26-1)/1 +1 = 26
        # output size after pooling = (26 - 2)/2 + 1 = 13
        # the output Tensor for one image: 
        #        will have the dimensions: (256, 13, 13) <= this data is important  
                                                                   
        # Fifth step: Flatten
        x = x.view(x.size(0), -1)
        
        # Sixth step: Dense (Linear), activation and dropout
        x = self.drop5(F.relu(self.fc1(x)))    
        
        # Seventh step: Dense (Linear), activation and dropout
        x = self.drop6(F.relu(self.fc2(x)))    
        
        # Eighth step (last): Final Dense (Linear)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, 
        # should be returned
        return x
