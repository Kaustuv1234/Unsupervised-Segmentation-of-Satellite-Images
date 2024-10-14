import torch
import torch.nn as nn
import torch.nn.functional as F



# # CNN model 5 channel
# class Model_BYOL(nn.Module):
#     def __init__(self,numberOfImageChannels, nFeaturesFinalLayer):
#         super(Model_BYOL, self).__init__()
        
#         self.Target = Model_Target(numberOfImageChannels, nFeaturesFinalLayer)
#         self.Online = Model_Online(numberOfImageChannels, nFeaturesFinalLayer)
#         # self.Predictor = Predictor(nFeaturesFinalLayer)

#     def forward(self, x_target, x_online):
#         x_target = self.Target(x_target)
#         x_online = self.Online(x_online)
#         # x_online = self.Predictor(x_online)
        
#         # return x_online, x_target
#         return x_target, x_online



# CNN model 5 channel
class Predictor(nn.Module):
    def __init__(self,nFeaturesFinalLayer):
        super(Predictor, self).__init__()
        
        # Prediction layer 7, 8
        self.conv7_online=nn.Conv2d(nFeaturesFinalLayer, 128, kernel_size=1, stride=1, padding=0)
        self.bn7_online=nn.BatchNorm2d(128)
        self.conv7_online.weight=torch.nn.init.kaiming_uniform_(self.conv7_online.weight)
        self.conv7_online.bias.data.fill_(0.01)
        self.bn7_online.bias.data.fill_(0.001)
        
        self.conv8_online=nn.Conv2d(128, nFeaturesFinalLayer, kernel_size=1, stride=1, padding=0)
        self.bn8_online=nn.BatchNorm2d(nFeaturesFinalLayer)
        self.conv8_online.weight=torch.nn.init.kaiming_uniform_(self.conv8_online.weight)
        self.conv8_online.bias.data.fill_(0.01)
        self.bn8_online.bias.data.fill_(0.001)
        
    def forward(self, x_online):


        # Prediction
        x_online=self.conv7_online(x_online)
        x_online=F.relu(x_online)
        x_online=self.bn7_online(x_online)  

        x_online=self.conv8_online(x_online)
        x_online=F.relu(x_online)
        x_online=self.bn8_online(x_online)  

        x_online=F.normalize(x_online)
        
        return x_online
        
 

# CNN model 5 channel
class Model_BYOL(nn.Module):
    def __init__(self,numberOfImageChannels, nFeaturesFinalLayer):
        super(Model_BYOL, self).__init__()
        
        self.model_online = Model5_Online_256(numberOfImageChannels, nFeaturesFinalLayer)
        self.model_target = Model5_Target_256(numberOfImageChannels, nFeaturesFinalLayer)
        # self.predictor = Predictor(nFeaturesFinalLayer)


    def forward(self, x_target, x_online):
        x_online = self.model_online(x_online) 
        x_target = self.model_target(x_target) 
        return x_target, x_online
        
        


        

# CNN model 5 channel
class Model5_Online_256(nn.Module):
    def __init__(self,numberOfImageChannels, nFeaturesFinalLayer):
        super(Model5_Online_256, self).__init__()
        
        # Representation layer 1
        self.conv1_online=nn.Conv2d(numberOfImageChannels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_online=nn.BatchNorm2d(64)
        self.conv1_online.weight=torch.nn.init.kaiming_uniform_(self.conv1_online.weight)
        self.conv1_online.bias.data.fill_(0.01)
        self.bn1_online.bias.data.fill_(0.001)
        

        # Representation layer 2
        self.conv2_online=nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) 
        self.bn2_online=nn.BatchNorm2d(128) 
        self.conv2_online.weight=torch.nn.init.kaiming_uniform_(self.conv2_online.weight)
        self.conv2_online.bias.data.fill_(0.01)
        self.bn2_online.bias.data.fill_(0.001)
        

        # Representation layer 3
        self.conv3_online=nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) 
        self.bn3_online=nn.BatchNorm2d(128)
        self.conv3_online.weight=torch.nn.init.kaiming_uniform_(self.conv3_online.weight)
        self.conv3_online.bias.data.fill_(0.01)
        self.bn3_online.bias.data.fill_(0.001)

        ##Conv layer 4
        self.conv3_1_online = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1) 
        self.bn3_1_online = nn.BatchNorm2d(256)
        self.conv3_1_online.weight = torch.nn.init.kaiming_uniform_(self.conv3_1_online.weight)
        self.conv3_1_online.bias.data.fill_(0.01)
        self.bn3_1_online.bias.data.fill_(0.001)

        ##Conv layer 5
        self.conv3_2_online = nn.Conv2d(256, 128, kernel_size = 3, stride = 1, padding = 1) 
        self.bn3_2_online = nn.BatchNorm2d(128)
        self.conv3_2_online.weight = torch.nn.init.kaiming_uniform_(self.conv3_2_online.weight)
        self.conv3_2_online.bias.data.fill_(0.01)
        self.bn3_2_online.bias.data.fill_(0.001)
        
        # Representation layer 4
        self.conv4_online=nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1) 
        self.bn4_online=nn.BatchNorm2d(64)
        self.conv4_online.weight=torch.nn.init.kaiming_uniform_(self.conv4_online.weight)
        self.conv4_online.bias.data.fill_(0.01)
        self.bn4_online.bias.data.fill_(0.001)


        # Projection layer 5, 6
        self.conv5_online=nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.bn5_online=nn.BatchNorm2d(128)
        self.conv5_online.weight=torch.nn.init.kaiming_uniform_(self.conv5_online.weight)
        self.conv5_online.bias.data.fill_(0.01)
        self.bn5_online.bias.data.fill_(0.001)
        
        self.conv6_online=nn.Conv2d(128, nFeaturesFinalLayer, kernel_size=1, stride=1, padding=0)
        self.bn6_online=nn.BatchNorm2d(nFeaturesFinalLayer)
        self.conv6_online.weight=torch.nn.init.kaiming_uniform_(self.conv6_online.weight)
        self.conv6_online.bias.data.fill_(0.01)
        self.bn6_online.bias.data.fill_(0.001)
       
        

        # Prediction layer 7, 8
        self.conv7_online=nn.Conv2d(nFeaturesFinalLayer, 128, kernel_size=1, stride=1, padding=0)
        self.bn7_online=nn.BatchNorm2d(128)
        self.conv7_online.weight=torch.nn.init.kaiming_uniform_(self.conv7_online.weight)
        self.conv7_online.bias.data.fill_(0.01)
        self.bn7_online.bias.data.fill_(0.001)
        
        self.conv8_online=nn.Conv2d(128, nFeaturesFinalLayer, kernel_size=1, stride=1, padding=0)
        self.bn8_online=nn.BatchNorm2d(nFeaturesFinalLayer)
        self.conv8_online.weight=torch.nn.init.kaiming_uniform_(self.conv8_online.weight)
        self.conv8_online.bias.data.fill_(0.01)
        self.bn8_online.bias.data.fill_(0.001)
        
        
        

    def forward(self, x_online):
        # Representation
        x_online=self.conv1_online(x_online)
        x_online=F.relu(x_online)
        x_online=self.bn1_online(x_online)

        x_online=self.conv2_online(x_online)
        x_online=F.relu(x_online)
        x_online=self.bn2_online(x_online)

        x_online=self.conv3_online(x_online)
        x_online=F.relu(x_online)
        x_online=self.bn3_online(x_online)

        x_online=self.conv3_1_online(x_online)
        x_online=F.relu(x_online)
        x_online=self.bn3_1_online(x_online)

        x_online=self.conv3_2_online(x_online)
        x_online=F.relu(x_online)
        x_online=self.bn3_2_online(x_online)

        x_online=self.conv4_online(x_online)
        x_online=F.relu(x_online)
        x_online=self.bn4_online(x_online)

        # Projection
        x_online=self.conv5_online(x_online)
        x_online=F.relu(x_online)
        x_online=self.bn5_online(x_online)  

        x_online=self.conv6_online(x_online)
        x_online=F.relu(x_online)
        x_online=self.bn6_online(x_online)  

        # Prediction
        x_online=self.conv7_online(x_online)
        x_online=F.relu(x_online)
        x_online=self.bn7_online(x_online)  

        x_online=self.conv8_online(x_online)
        x_online=F.relu(x_online)
        x_online=self.bn8_online(x_online)  

        x_online=F.normalize(x_online)
        
        return x_online
        
        




# CNN model 5 channel
class Model5_Target_256(nn.Module):
    def __init__(self,numberOfImageChannels, nFeaturesFinalLayer):
        super(Model5_Target_256, self).__init__()
        
        # Representation layer 1
        self.conv1_target=nn.Conv2d(numberOfImageChannels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_target=nn.BatchNorm2d(64)
        self.conv1_target.weight=torch.nn.init.kaiming_uniform_(self.conv1_target.weight)
        self.conv1_target.bias.data.fill_(0.01)
        self.bn1_target.bias.data.fill_(0.001)

        
        # Representation layer 2
        self.conv2_target=nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) 
        self.bn2_target=nn.BatchNorm2d(128) 
        self.conv2_target.weight=torch.nn.init.kaiming_uniform_(self.conv2_target.weight)
        self.conv2_target.bias.data.fill_(0.01)
        self.bn2_target.bias.data.fill_(0.001)
        

        # # Representation layer 3
        self.conv3_target=nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) 
        self.bn3_target=nn.BatchNorm2d(128)
        self.conv3_target.weight=torch.nn.init.kaiming_uniform_(self.conv3_target.weight)
        self.conv3_target.bias.data.fill_(0.01)
        self.bn3_target.bias.data.fill_(0.001)
        
        ##Conv layer 4
        self.conv3_1_target = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1) 
        self.bn3_1_target = nn.BatchNorm2d(256)
        self.conv3_1_target.weight = torch.nn.init.kaiming_uniform_(self.conv3_1_target.weight)
        self.conv3_1_target.bias.data.fill_(0.01)
        self.bn3_1_target.bias.data.fill_(0.001)

        ##Conv layer 5
        self.conv3_2_target = nn.Conv2d(256, 128, kernel_size = 3, stride = 1, padding = 1) 
        self.bn3_2_target = nn.BatchNorm2d(128)
        self.conv3_2_target.weight = torch.nn.init.kaiming_uniform_(self.conv3_2_target.weight)
        self.conv3_2_target.bias.data.fill_(0.01)
        self.bn3_2_target.bias.data.fill_(0.001)
        
        # Representation layer 4
        self.conv4_target=nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1) 
        self.bn4_target=nn.BatchNorm2d(64)
        self.conv4_target.weight=torch.nn.init.kaiming_uniform_(self.conv4_target.weight)
        self.conv4_target.bias.data.fill_(0.01)
        self.bn4_target.bias.data.fill_(0.001)
        
        
        # Projection layer 5, 6
        self.conv5_target=nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.bn5_target=nn.BatchNorm2d(128)
        self.conv5_target.weight=torch.nn.init.kaiming_uniform_(self.conv5_target.weight)
        self.conv5_target.bias.data.fill_(0.01)
        self.bn5_target.bias.data.fill_(0.001)
    
        self.conv6_target=nn.Conv2d(128, nFeaturesFinalLayer, kernel_size=1, stride=1, padding=0)
        self.bn6_target=nn.BatchNorm2d(nFeaturesFinalLayer)
        self.conv6_target.weight=torch.nn.init.kaiming_uniform_(self.conv6_target.weight)
        self.conv6_target.bias.data.fill_(0.01)
        self.bn6_target.bias.data.fill_(0.001)
        

        
        
        

    def forward(self, x_target):
        # Representation
        x_target=self.conv1_target(x_target)
        x_target=F.relu(x_target)
        x_target=self.bn1_target(x_target)

        x_target=self.conv2_target(x_target)
        x_target=F.relu(x_target)
        x_target=self.bn2_target(x_target)

        x_target=self.conv3_target(x_target)
        x_target=F.relu(x_target)
        x_target=self.bn3_target(x_target)

        x_target=self.conv3_1_target(x_target)
        x_target=F.relu(x_target)
        x_target=self.bn3_1_target(x_target)

        x_target=self.conv3_2_target(x_target)
        x_target=F.relu(x_target)
        x_target=self.bn3_2_target(x_target)

        x_target=self.conv4_target(x_target)
        x_target=F.relu(x_target)
        x_target=self.bn4_target(x_target)

        # Projection
        x_target=self.conv5_target(x_target)
        x_target=F.relu(x_target)
        x_target=self.bn5_target(x_target)  
        x_target=self.conv6_target(x_target)
        x_target=F.relu(x_target)
        x_target=self.bn6_target(x_target)   
        x_target=F.normalize(x_target)
        
        return x_target
        