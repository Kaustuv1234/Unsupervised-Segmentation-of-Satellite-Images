import torch
import torch.nn as nn
import torch.nn.functional as F






# CNN model 5 channel
class Model5(nn.Module):
    def __init__(self,numberOfImageChannels,nFeaturesFinalLayer):
        super(Model5, self).__init__()
               
        kernelSize = 3
        paddingSize = int((kernelSize-1)/2)

        
        ##Conv layer 1
        self.conv1_Stream1 = nn.Conv2d(numberOfImageChannels, 64, kernel_size = kernelSize, stride = 1, padding = paddingSize)
        self.bn1_Stream1 = nn.BatchNorm2d(64)
        self.conv1_Stream1.weight = torch.nn.init.kaiming_uniform_(self.conv1_Stream1.weight)
        self.conv1_Stream1.bias.data.fill_(0.01)
        self.bn1_Stream1.bias.data.fill_(0.001)
        
        
        self.conv1_Stream2 = nn.Conv2d(numberOfImageChannels, 64, kernel_size = kernelSize, stride = 1, padding = paddingSize)
        self.bn1_Stream2 = nn.BatchNorm2d(64)
        self.conv1_Stream2.weight = torch.nn.init.kaiming_uniform_(self.conv1_Stream2.weight)
        self.conv1_Stream2.bias.data.fill_(0.01)
        self.bn1_Stream2.bias.data.fill_(0.001)
        
        ##Conv layer 2
        self.conv2_Stream1 = nn.Conv2d(64, 128, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn2_Stream1 = nn.BatchNorm2d(128) 
        self.conv2_Stream1.weight = torch.nn.init.kaiming_uniform_(self.conv2_Stream1.weight)
        self.conv2_Stream1.bias.data.fill_(0.01)
        self.bn2_Stream1.bias.data.fill_(0.001)
        
        
        self.conv2_Stream2 = nn.Conv2d(64, 128, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn2_Stream2 = nn.BatchNorm2d(128) 
        self.conv2_Stream2.weight = torch.nn.init.kaiming_uniform_(self.conv2_Stream2.weight)
        self.conv2_Stream2.bias.data.fill_(0.01)
        self.bn2_Stream2.bias.data.fill_(0.001)
        
        ##Conv layer 3
        self.conv3_Stream1 = nn.Conv2d(128, 128, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn3_Stream1 = nn.BatchNorm2d(128)
        self.conv3_Stream1.weight = torch.nn.init.kaiming_uniform_(self.conv3_Stream1.weight)
        self.conv3_Stream1.bias.data.fill_(0.01)
        self.bn3_Stream1.bias.data.fill_(0.001)
        
        self.conv3_Stream2 = nn.Conv2d(128, 128, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn3_Stream2 = nn.BatchNorm2d(128)
        self.conv3_Stream2.weight = torch.nn.init.kaiming_uniform_(self.conv3_Stream2.weight)
        self.conv3_Stream2.bias.data.fill_(0.01)
        self.bn3_Stream2.bias.data.fill_(0.001)
        
        
        
        ##Conv layer 4
        self.conv4_Stream1 = nn.Conv2d(128, 64, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn4_Stream1 = nn.BatchNorm2d(64)
        self.conv4_Stream1.weight = torch.nn.init.kaiming_uniform_(self.conv4_Stream1.weight)
        self.conv4_Stream1.bias.data.fill_(0.01)
        self.bn4_Stream1.bias.data.fill_(0.001)
        
        
        self.conv4_Stream2 = nn.Conv2d(128, 64, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn4_Stream2 = nn.BatchNorm2d(64)
        self.conv4_Stream2.weight = torch.nn.init.kaiming_uniform_(self.conv4_Stream2.weight)
        self.conv4_Stream2.bias.data.fill_(0.01)
        self.bn4_Stream2.bias.data.fill_(0.001)
        
        
         
        ##Conv layer 5
        self.conv5 = nn.Conv2d(64, nFeaturesFinalLayer, kernel_size = 1, stride = 1, padding = 0)
        self.bn5 = nn.BatchNorm2d(nFeaturesFinalLayer)
        self.conv5.weight = torch.nn.init.kaiming_uniform_(self.conv5.weight)
        self.conv5.bias.data.fill_(0.01)
        self.bn5.bias.data.fill_(0.001)
        

    def forward(self, x_Stream1, x_Stream2):
        x_Stream1 = self.conv1_Stream1(x_Stream1)
        x_Stream1 = F.relu(x_Stream1)
        x_Stream1 = self.bn1_Stream1(x_Stream1)
        x_Stream1 = self.conv2_Stream1(x_Stream1)
        x_Stream1 = F.relu(x_Stream1)
        x_Stream1 = self.bn2_Stream1(x_Stream1)
        x_Stream1 = self.conv3_Stream1(x_Stream1)
        x_Stream1 = F.relu(x_Stream1)
        x_Stream1 = self.bn3_Stream1(x_Stream1)
        x_Stream1 = self.conv4_Stream1(x_Stream1)
        x_Stream1 = F.relu(x_Stream1)
        x_Stream1 = self.bn4_Stream1(x_Stream1)
        x_Stream1 = self.conv5(x_Stream1)
        x_Stream1 = self.bn5(x_Stream1)  ##Note there is no relu between last conv and bn
        
        x_Stream2 = self.conv1_Stream2(x_Stream2)
        x_Stream2 = F.relu(x_Stream2)
        x_Stream2 = self.bn1_Stream2(x_Stream2)
        x_Stream2 = self.conv2_Stream2(x_Stream2)
        x_Stream2 = F.relu(x_Stream2)
        x_Stream2 = self.bn2_Stream2(x_Stream2)
        x_Stream2 = self.conv3_Stream2(x_Stream2)
        x_Stream2 = F.relu(x_Stream2)
        x_Stream2 = self.bn3_Stream2(x_Stream2)
        x_Stream2 = self.conv4_Stream2(x_Stream2)
        x_Stream2 = F.relu(x_Stream2)
        x_Stream2 = self.bn4_Stream2(x_Stream2)
        x_Stream2 = self.conv5(x_Stream2)
        x_Stream2 = self.bn5(x_Stream2)  ##Note there is no relu between last conv and bn
        
        return x_Stream1, x_Stream2
        



# CNN model 5 channel
class Model5_maxpool(nn.Module):
    def __init__(self,numberOfImageChannels,nFeaturesFinalLayer):
        super(Model5_maxpool, self).__init__()
        
        ##see this page: https://discuss.pytorch.org/t/extracting-and-using-features-from-a-pretrained-model/20723
        #pretrainedModel = vgg16(pretrained = True)
        #self.Vgg16features = nn.Sequential(
         #           *list(pretrainedModel.features.children())[:3])
        
        kernelSize = 3
        paddingSize = int((kernelSize-1)/2)

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ##Conv layer 1
        self.conv1_Stream1 = nn.Conv2d(numberOfImageChannels, 64, kernel_size = kernelSize, stride = 1, padding = paddingSize)
        self.bn1_Stream1 = nn.BatchNorm2d(64)
        self.conv1_Stream1.weight = torch.nn.init.kaiming_uniform_(self.conv1_Stream1.weight)
        self.conv1_Stream1.bias.data.fill_(0.01)
        #self.bn1.weight.data.fill_(1)
        self.bn1_Stream1.bias.data.fill_(0.001)
        
        
        self.conv1_Stream2 = nn.Conv2d(numberOfImageChannels, 64, kernel_size = kernelSize, stride = 1, padding = paddingSize)
        self.bn1_Stream2 = nn.BatchNorm2d(64)
        self.conv1_Stream2.weight = torch.nn.init.kaiming_uniform_(self.conv1_Stream2.weight)
        self.conv1_Stream2.bias.data.fill_(0.01)
        #self.bn1.weight.data.fill_(1)
        self.bn1_Stream2.bias.data.fill_(0.001)
        
        ##Conv layer 2
        self.conv2_Stream1 = nn.Conv2d(64, 128, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn2_Stream1 = nn.BatchNorm2d(128) 
        self.conv2_Stream1.weight = torch.nn.init.kaiming_uniform_(self.conv2_Stream1.weight)
        self.conv2_Stream1.bias.data.fill_(0.01)
        #self.bn2_Stream1.weight.data.fill_(1)
        self.bn2_Stream1.bias.data.fill_(0.001)
        
        
        self.conv2_Stream2 = nn.Conv2d(64, 128, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn2_Stream2 = nn.BatchNorm2d(128) 
        self.conv2_Stream2.weight = torch.nn.init.kaiming_uniform_(self.conv2_Stream2.weight)
        self.conv2_Stream2.bias.data.fill_(0.01)
        #self.bn2_Stream1.weight.data.fill_(1)
        self.bn2_Stream2.bias.data.fill_(0.001)
        
        ##Conv layer 3
        self.conv3_Stream1 = nn.Conv2d(128, 128, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn3_Stream1 = nn.BatchNorm2d(128)
        self.conv3_Stream1.weight = torch.nn.init.kaiming_uniform_(self.conv3_Stream1.weight)
        self.conv3_Stream1.bias.data.fill_(0.01)
        #self.bn3_Stream1.weight.data.fill_(1)
        self.bn3_Stream1.bias.data.fill_(0.001)
        
        self.conv3_Stream2 = nn.Conv2d(128, 128, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn3_Stream2 = nn.BatchNorm2d(128)
        self.conv3_Stream2.weight = torch.nn.init.kaiming_uniform_(self.conv3_Stream2.weight)
        self.conv3_Stream2.bias.data.fill_(0.01)
        #self.bn3_Stream1.weight.data.fill_(1)
        self.bn3_Stream2.bias.data.fill_(0.001)
        
        
        
        ##Conv layer 4
        self.conv4_Stream1 = nn.Conv2d(128, 64, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn4_Stream1 = nn.BatchNorm2d(64)
        self.conv4_Stream1.weight = torch.nn.init.kaiming_uniform_(self.conv4_Stream1.weight)
        self.conv4_Stream1.bias.data.fill_(0.01)
        #self.bn3_Stream1.weight.data.fill_(1)
        self.bn4_Stream1.bias.data.fill_(0.001)
        
        
        self.conv4_Stream2 = nn.Conv2d(128, 64, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn4_Stream2 = nn.BatchNorm2d(64)
        self.conv4_Stream2.weight = torch.nn.init.kaiming_uniform_(self.conv4_Stream2.weight)
        self.conv4_Stream2.bias.data.fill_(0.01)
        #self.bn4_Stream1.weight.data.fill_(1)
        self.bn4_Stream2.bias.data.fill_(0.001)
        
        
         
        ##Conv layer 5
        self.conv5 = nn.Conv2d(64, nFeaturesFinalLayer, kernel_size = 1, stride = 1, padding = 0)
        self.bn5 = nn.BatchNorm2d(nFeaturesFinalLayer)
        self.conv5.weight = torch.nn.init.kaiming_uniform_(self.conv5.weight)
        self.conv5.bias.data.fill_(0.01)
        #self.bn5.weight.data.fill_(1)
        self.bn5.bias.data.fill_(0.001)
        

    def forward(self, x_Stream1, x_Stream2):
        #x = self.Vgg16features(x)
        x_Stream1 = self.conv1_Stream1(x_Stream1)
        x_Stream1 = F.relu(x_Stream1)
        x_Stream1 = self.bn1_Stream1(x_Stream1)
        x_Stream1 = self.conv2_Stream1(x_Stream1)
        x_Stream1 = F.relu(x_Stream1)
        x_Stream1 = self.bn2_Stream1(x_Stream1)
        x_Stream1 = self.max_pool2d(x_Stream1)
        x_Stream1 = self.conv3_Stream1(x_Stream1)
        x_Stream1 = F.relu(x_Stream1)
        x_Stream1 = self.bn3_Stream1(x_Stream1)
        x_Stream1 = self.conv4_Stream1(x_Stream1)
        x_Stream1 = F.relu(x_Stream1)
        x_Stream1 = self.bn4_Stream1(x_Stream1)
        x_Stream1 = self.conv5(x_Stream1)
        x_Stream1 = self.bn5(x_Stream1)  ##Note there is no relu between last conv and bn
        
        x_Stream2 = self.conv1_Stream2(x_Stream2)
        x_Stream2 = F.relu(x_Stream2)
        x_Stream2 = self.bn1_Stream2(x_Stream2)
        x_Stream2 = self.conv2_Stream2(x_Stream2)
        x_Stream2 = F.relu(x_Stream2)
        x_Stream2 = self.bn2_Stream2(x_Stream2)
        x_Stream2 = self.max_pool2d(x_Stream2)
        x_Stream2 = self.conv3_Stream2(x_Stream2)
        x_Stream2 = F.relu(x_Stream2)
        x_Stream2 = self.bn3_Stream2(x_Stream2)
        x_Stream2 = self.conv4_Stream2(x_Stream2)
        x_Stream2 = F.relu(x_Stream2)
        x_Stream2 = self.bn4_Stream2(x_Stream2)
        x_Stream2 = self.conv5(x_Stream2)
        x_Stream2 = self.bn5(x_Stream2)  ##Note there is no relu between last conv and bn
        
        return x_Stream1, x_Stream2
        
        
        
        

# CNN model 4 channel
class Model4(nn.Module):
    def __init__(self,numberOfImageChannels,nFeaturesFinalLayer):
        super(Model4, self).__init__()
        
        ##see this page: https://discuss.pytorch.org/t/extracting-and-using-features-from-a-pretrained-model/20723
        #pretrainedModel = vgg16(pretrained = True)
        #self.Vgg16features = nn.Sequential(
         #           *list(pretrainedModel.features.children())[:3])
        
        kernelSize = 3
        paddingSize = int((kernelSize-1)/2)
        
        ##Conv layer 1
        self.conv1_Stream1 = nn.Conv2d(numberOfImageChannels, 64, kernel_size = kernelSize, stride = 1, padding = paddingSize)
        self.bn1_Stream1 = nn.BatchNorm2d(64)
        self.conv1_Stream1.weight = torch.nn.init.kaiming_uniform_(self.conv1_Stream1.weight)
        self.conv1_Stream1.bias.data.fill_(0.01)
        #self.bn1.weight.data.fill_(1)
        self.bn1_Stream1.bias.data.fill_(0.001)
        
        
        self.conv1_Stream2 = nn.Conv2d(numberOfImageChannels, 64, kernel_size = kernelSize, stride = 1, padding = paddingSize)
        self.bn1_Stream2 = nn.BatchNorm2d(64)
        self.conv1_Stream2.weight = torch.nn.init.kaiming_uniform_(self.conv1_Stream2.weight)
        self.conv1_Stream2.bias.data.fill_(0.01)
        #self.bn1.weight.data.fill_(1)
        self.bn1_Stream2.bias.data.fill_(0.001)
        
        ##Conv layer 2
        self.conv2_Stream1 = nn.Conv2d(64, 128, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn2_Stream1 = nn.BatchNorm2d(128) 
        self.conv2_Stream1.weight = torch.nn.init.kaiming_uniform_(self.conv2_Stream1.weight)
        self.conv2_Stream1.bias.data.fill_(0.01)
        #self.bn2_Stream1.weight.data.fill_(1)
        self.bn2_Stream1.bias.data.fill_(0.001)
        
        
        self.conv2_Stream2 = nn.Conv2d(64, 128, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn2_Stream2 = nn.BatchNorm2d(128) 
        self.conv2_Stream2.weight = torch.nn.init.kaiming_uniform_(self.conv2_Stream2.weight)
        self.conv2_Stream2.bias.data.fill_(0.01)
        #self.bn2_Stream1.weight.data.fill_(1)
        self.bn2_Stream2.bias.data.fill_(0.001)
        
        ##Conv layer 3
        self.conv3_Stream1 = nn.Conv2d(128, 64, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn3_Stream1 = nn.BatchNorm2d(64)
        self.conv3_Stream1.weight = torch.nn.init.kaiming_uniform_(self.conv3_Stream1.weight)
        self.conv3_Stream1.bias.data.fill_(0.01)
        #self.bn3_Stream1.weight.data.fill_(1)
        self.bn3_Stream1.bias.data.fill_(0.001)
        
        self.conv3_Stream2 = nn.Conv2d(128, 64, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn3_Stream2 = nn.BatchNorm2d(64)
        self.conv3_Stream2.weight = torch.nn.init.kaiming_uniform_(self.conv3_Stream2.weight)
        self.conv3_Stream2.bias.data.fill_(0.01)
        #self.bn3_Stream1.weight.data.fill_(1)
        self.bn3_Stream2.bias.data.fill_(0.001)
        
        
         
        ##Conv layer 4
        self.conv4 = nn.Conv2d(64, nFeaturesFinalLayer, kernel_size = 1, stride = 1, padding = 0)
        self.bn4 = nn.BatchNorm2d(nFeaturesFinalLayer)
        self.conv4.weight = torch.nn.init.kaiming_uniform_(self.conv4.weight)
        self.conv4.bias.data.fill_(0.01)
        self.bn4.bias.data.fill_(0.001)
        

    def forward(self, x_Stream1, x_Stream2):
        #x = self.Vgg16features(x)
        x_Stream1 = self.conv1_Stream1(x_Stream1)
        x_Stream1 = F.relu(x_Stream1)
        x_Stream1 = self.bn1_Stream1(x_Stream1)
        x_Stream1 = self.conv2_Stream1(x_Stream1)
        x_Stream1 = F.relu(x_Stream1)
        x_Stream1 = self.bn2_Stream1(x_Stream1)
        x_Stream1 = self.conv3_Stream1(x_Stream1)
        x_Stream1 = F.relu(x_Stream1)
        x_Stream1 = self.bn3_Stream1(x_Stream1)
        x_Stream1 = self.conv4(x_Stream1)
        x_Stream1 = self.bn4(x_Stream1)  ##Note there is no relu between last conv and bn
        
        x_Stream2 = self.conv1_Stream2(x_Stream2)
        x_Stream2 = F.relu(x_Stream2)
        x_Stream2 = self.bn1_Stream2(x_Stream2)
        x_Stream2 = self.conv2_Stream2(x_Stream2)
        x_Stream2 = F.relu(x_Stream2)
        x_Stream2 = self.bn2_Stream2(x_Stream2)
        x_Stream2 = self.conv3_Stream2(x_Stream2)
        x_Stream2 = F.relu(x_Stream2)
        x_Stream2 = self.bn3_Stream2(x_Stream2)
        x_Stream2 = self.conv4(x_Stream2)
        x_Stream2 = self.bn4(x_Stream2)  ##Note there is no relu between last conv and bn
        
        return x_Stream1, x_Stream2



# CNN model 6 channel
class Model6(nn.Module):
    def __init__(self,numberOfImageChannels,nFeaturesFinalLayer, initialization=True):
        super(Model6, self).__init__()
        
        kernelSize = 3
        paddingSize = int((kernelSize-1)/2)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ##Conv layer 1
        self.conv1_Stream1 = nn.Conv2d(numberOfImageChannels, 64, kernel_size = kernelSize, stride = 1, padding = paddingSize)
        self.bn1_Stream1 = nn.BatchNorm2d(64)
        if initialization:
            self.conv1_Stream1.weight = torch.nn.init.kaiming_uniform_(self.conv1_Stream1.weight)
            self.conv1_Stream1.bias.data.fill_(0.01)
            self.bn1_Stream1.bias.data.fill_(0.001)
        
        
        self.conv1_Stream2 = nn.Conv2d(numberOfImageChannels, 64, kernel_size = kernelSize, stride = 1, padding = paddingSize)
        self.bn1_Stream2 = nn.BatchNorm2d(64)
        if initialization:
            self.conv1_Stream2.weight = torch.nn.init.kaiming_uniform_(self.conv1_Stream2.weight)
            self.conv1_Stream2.bias.data.fill_(0.01)
            self.bn1_Stream2.bias.data.fill_(0.001)
        
        ##Conv layer 2
        self.conv2_Stream1 = nn.Conv2d(64, 128, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn2_Stream1 = nn.BatchNorm2d(128) 
        if initialization:
            self.conv2_Stream1.weight = torch.nn.init.kaiming_uniform_(self.conv2_Stream1.weight)
            self.conv2_Stream1.bias.data.fill_(0.01)
            self.bn2_Stream1.bias.data.fill_(0.001)
        
        
        self.conv2_Stream2 = nn.Conv2d(64, 128, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn2_Stream2 = nn.BatchNorm2d(128) 
        if initialization:
            self.conv2_Stream2.weight = torch.nn.init.kaiming_uniform_(self.conv2_Stream2.weight)
            self.conv2_Stream2.bias.data.fill_(0.01)
            self.bn2_Stream2.bias.data.fill_(0.001)
            
        ##Conv layer 3
        self.conv3_Stream1 = nn.Conv2d(128, 256, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn3_Stream1 = nn.BatchNorm2d(256)
        if initialization:
            self.conv3_Stream1.weight = torch.nn.init.kaiming_uniform_(self.conv3_Stream1.weight)
            self.conv3_Stream1.bias.data.fill_(0.01)
            self.bn3_Stream1.bias.data.fill_(0.001)
        
        self.conv3_Stream2 = nn.Conv2d(128, 256, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn3_Stream2 = nn.BatchNorm2d(256)
        if initialization:
            self.conv3_Stream2.weight = torch.nn.init.kaiming_uniform_(self.conv3_Stream2.weight)
            self.conv3_Stream2.bias.data.fill_(0.01)
            self.bn3_Stream2.bias.data.fill_(0.001)
        
        
        
        ##Conv layer 4
        self.conv4_Stream1 = nn.Conv2d(256, 128, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn4_Stream1 = nn.BatchNorm2d(128)
        if initialization:
            self.conv4_Stream1.weight = torch.nn.init.kaiming_uniform_(self.conv4_Stream1.weight)
            self.conv4_Stream1.bias.data.fill_(0.01)
            self.bn4_Stream1.bias.data.fill_(0.001)
        
        
        self.conv4_Stream2 = nn.Conv2d(256, 128, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn4_Stream2 = nn.BatchNorm2d(128)
        if initialization:
            self.conv4_Stream2.weight = torch.nn.init.kaiming_uniform_(self.conv4_Stream2.weight)
            self.conv4_Stream2.bias.data.fill_(0.01)
            self.bn4_Stream2.bias.data.fill_(0.001)
        
        
        ##Conv layer 5
        self.conv5_Stream1 = nn.Conv2d(128, 64, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn5_Stream1 = nn.BatchNorm2d(64)
        if initialization:
            self.conv5_Stream1.weight = torch.nn.init.kaiming_uniform_(self.conv5_Stream1.weight)
            self.conv5_Stream1.bias.data.fill_(0.01)
            self.bn5_Stream1.bias.data.fill_(0.001)
            
        
        self.conv5_Stream2 = nn.Conv2d(128, 64, kernel_size = kernelSize, stride = 1, padding = paddingSize) 
        self.bn5_Stream2 = nn.BatchNorm2d(64)
        if initialization:
            self.conv5_Stream2.weight = torch.nn.init.kaiming_uniform_(self.conv5_Stream2.weight)
            self.conv5_Stream2.bias.data.fill_(0.01)
            self.bn5_Stream2.bias.data.fill_(0.001)
            
        
         
        ##Conv layer 6
        self.conv6 = nn.Conv2d(64, nFeaturesFinalLayer, kernel_size = 1, stride = 1, padding = 0)
        self.bn6 = nn.BatchNorm2d(nFeaturesFinalLayer)
        if initialization:
            self.conv6.weight = torch.nn.init.kaiming_uniform_(self.conv6.weight)
            self.conv6.bias.data.fill_(0.01)
            self.bn6.bias.data.fill_(0.001)
        

    def forward(self, x_Stream1, x_Stream2):
        x_Stream1 = self.conv1_Stream1(x_Stream1)
        x_Stream1 = F.relu(x_Stream1)
        x_Stream1 = self.bn1_Stream1(x_Stream1)
        x_Stream1 = self.conv2_Stream1(x_Stream1)
        x_Stream1 = F.relu(x_Stream1)
        x_Stream1 = self.bn2_Stream1(x_Stream1)
        # x_Stream1 = self.max_pool2d(x_Stream1)
        x_Stream1 = self.conv3_Stream1(x_Stream1)
        x_Stream1 = F.relu(x_Stream1)
        x_Stream1 = self.bn3_Stream1(x_Stream1)
        # x_Stream1 = self.max_pool2d(x_Stream1)
        x_Stream1 = self.conv4_Stream1(x_Stream1)
        x_Stream1 = F.relu(x_Stream1)
        x_Stream1 = self.bn4_Stream1(x_Stream1)
        x_Stream1 = self.conv5_Stream1(x_Stream1)
        x_Stream1 = F.relu(x_Stream1)
        x_Stream1 = self.bn5_Stream1(x_Stream1)
        x_Stream1 = self.conv6(x_Stream1)
        x_Stream1 = self.bn6(x_Stream1)  ##Note there is no relu between last conv and bn
        
        x_Stream2 = self.conv1_Stream2(x_Stream2)
        x_Stream2 = F.relu(x_Stream2)
        x_Stream2 = self.bn1_Stream2(x_Stream2)
        x_Stream2 = self.conv2_Stream2(x_Stream2)
        x_Stream2 = F.relu(x_Stream2)
        x_Stream2 = self.bn2_Stream2(x_Stream2)
        # x_Stream2 = self.max_pool2d(x_Stream2)
        x_Stream2 = self.conv3_Stream2(x_Stream2)
        x_Stream2 = F.relu(x_Stream2)
        x_Stream2 = self.bn3_Stream2(x_Stream2)
        # x_Stream2 = self.max_pool2d(x_Stream2)
        x_Stream2 = self.conv4_Stream2(x_Stream2)
        x_Stream2 = F.relu(x_Stream2)
        x_Stream2 = self.bn4_Stream2(x_Stream2)
        x_Stream2 = self.conv5_Stream2(x_Stream2)
        x_Stream2 = F.relu(x_Stream2)
        x_Stream2 = self.bn5_Stream2(x_Stream2)
        x_Stream2 = self.conv6(x_Stream2)
        x_Stream2 = self.bn6(x_Stream2)  ##Note there is no relu between last conv and bn
        
        return x_Stream1, x_Stream2
