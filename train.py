import os
import sys
import torch
import torchvision
import numpy as np

import wandb

from math import ceil
import argparse
from tqdm import tqdm

from networksVaryingKernel import *
from networksVaryingKernel_BYOL import *
from utilities import *
from dataHandling import *



##Defining Parameters
parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--manualSeed', type = int, default = 85, help = 'manual seed')
parser.add_argument('--dataset', default='Vaihingen', choices=['Potsdam_RGB', 'Potsdam_RGBIR', 'Vaihingen'], help='choose dataset') 
parser.add_argument('--nFeaturesFinalLayer', type = int, default = 8, help = 'Number of features in the final classification layer')
parser.add_argument('--no_patches', type = int, default = 690, help = 'Number of training epochs')
parser.add_argument('--trainingBatchSize', type = int, default = 6, help = 'batchsize')
parser.add_argument('--trainingPatchSize', type = int, default = 448, help = 'image patch size')
parser.add_argument('--trainingStrideSize', type = int, default = 448, help = 'stride between patches')
parser.add_argument('--modelName', type = str, default = 'Model6', choices=['Model_BYOL', 'Model6'], help = 'Model name')
args = parser.parse_args()
print(args)

torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)
np.random.seed(args.manualSeed)



# saveModelPath = './trainedModels/' + args.dataset + '_Model.pth'
saveModelPath = f'./trainedModels/{args.dataset}_{args.modelName}_{args.nFeaturesFinalLayer}.pth'

numberOfImageChannels = 3
if args.dataset == 'Potsdam_RGBIR':
    numberOfImageChannels = 4
    args.trainingBatchSize = 4




if args.modelName == 'Model_BYOL':
    model = Model_BYOL(numberOfImageChannels, args.nFeaturesFinalLayer)
    online_lr = 0.001
    target_lr = 0.002
    optimizer = torch.optim.SGD([{"params": model.model_target.parameters(), "lr": 0.002},
                             {"params": model.model_online.parameters(), "lr": 0.001}],
                              momentum = 0.9, 
                              weight_decay=0.001)
    # optimizer = torch.optim.SGD([{"params": model.model_target.parameters(), "lr": 0.001},
    #                          {"params": model.model_online.conv1_online.parameters(), "lr": 0.001},
    #                         {"params": model.model_online.bn1_online.parameters(), "lr": 0.001},
    #                         {"params": model.model_online.conv2_online.parameters(), "lr": 0.001},
    #                         {"params": model.model_online.bn2_online.parameters(), "lr": 0.001},
    #                         {"params": model.model_online.conv3_online.parameters(), "lr": 0.001},
    #                         {"params": model.model_online.bn3_online.parameters(), "lr": 0.001},
    #                         {"params": model.model_online.conv3_1_online.parameters(), "lr": 0.001},
    #                         {"params": model.model_online.bn3_1_online.parameters(), "lr": 0.001},
    #                         {"params": model.model_online.conv3_2_online.parameters(), "lr": 0.001},
    #                         {"params": model.model_online.bn3_2_online.parameters(), "lr": 0.001},
    #                         {"params": model.model_online.conv4_online.parameters(), "lr": 0.001},
    #                         {"params": model.model_online.bn4_online.parameters(), "lr": 0.001},
    #                         {"params": model.model_online.conv5_online.parameters(), "lr": 0.001},
    #                         {"params": model.model_online.bn5_online.parameters(), "lr": 0.001},
    #                         {"params": model.model_online.conv6_online.parameters(), "lr": 0.001},
    #                         {"params": model.model_online.bn6_online.parameters(), "lr": 0.001},
    #                         {"params": model.model_online.conv7_online.parameters(), "lr": 0.003},
    #                         {"params": model.model_online.bn7_online.parameters(), "lr": 0.003},
    #                         {"params": model.model_online.conv8_online.parameters(), "lr": 0.003},
    #                         {"params": model.model_online.bn8_online.parameters(), "lr": 0.003}],
    #                           momentum = 0.9, 
    #                           weight_decay=0.001)
elif args.modelName == 'Model6':
    model = Model6(numberOfImageChannels, args.nFeaturesFinalLayer)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9, weight_decay=0.001) ##Adam or SGD
else:
    sys.exit('Unrecognized model name')


model.cuda()
model.train()

lossFunction = torch.nn.CrossEntropyLoss()
lossFunctionSecondary = torch.nn.MSELoss()
'''optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.001)  ##Adam or SGD

slow_lr = 0.001
fast_lr = 0.002
optimizer = torch.optim.SGD([{"params": model.conv1_online.parameters(), "lr": slow_lr},
                             {"params": model.bn1_online.parameters(), "lr": slow_lr},
                             {"params": model.conv2_online.parameters(), "lr": slow_lr},
                             {"params": model.bn2_online.parameters(), "lr": slow_lr},
                             {"params": model.conv3_online.parameters(), "lr": slow_lr},
                             {"params": model.bn3_online.parameters(), "lr": slow_lr},
                             {"params": model.conv4_online.parameters(), "lr": slow_lr},
                             {"params": model.bn4_online.parameters(), "lr": slow_lr}, 
                             {"params": model.conv5_online.parameters(), "lr": slow_lr},
                             {"params": model.bn5_online.parameters(), "lr": slow_lr}, 

                             {"params": model.conv1_Stream2.parameters(), "lr": fast_lr},
                             {"params": model.bn1_Stream2.parameters(), "lr": fast_lr},
                             {"params": model.conv2_Stream2.parameters(), "lr": fast_lr},
                             {"params": model.bn2_Stream2.parameters(), "lr": fast_lr},
                             {"params": model.conv3_Stream2.parameters(), "lr": fast_lr},
                             {"params": model.bn3_Stream2.parameters(), "lr": fast_lr},
                             {"params": model.conv4_Stream2.parameters(), "lr": fast_lr},
                             {"params": model.bn4_Stream2.parameters(), "lr": fast_lr},
                             {"params": model.conv5_Stream2.parameters(), "lr": fast_lr},
                             {"params": model.bn5_Stream2.parameters(), "lr": fast_lr},

                             {"params": model.conv6.parameters(), "lr": slow_lr},
                             {"params": model.bn6.parameters(), "lr": slow_lr}],
                              momentum = 0.9,
                              weight_decay=0.001)'''

  

if args.dataset == 'Vaihingen':
    test_scene_ids = ['11', '15', '28', '30', '34']
    patch_path = '../datasets/Vaihingen/Vaihingen_448/img'
    imagePath = '../datasets/Vaihingen/Vaihingen_tiff/img/top_mosaic_09cm_area1.tif'
elif args.dataset == 'Potsdam_RGB':
    test_scene_ids = ['6_9', '6_14', '4_13', '2_10', '7_10', '2_11', '7_13']
    patch_path = '../datasets/Potsdam/Potsdam_RGB_1000'
    imagePath = '../datasets/Potsdam/Potsdam_tiff/2_Ortho_RGB/top_potsdam_7_7_RGB.tif'
elif args.dataset == 'Potsdam_RGBIR':
    test_scene_ids = ['6_9', '6_14', '4_13', '2_10', '7_10', '2_11', '7_13']
    patch_path = '../datasets/Potsdam/Potsdam_RGBIR_1000'
    imagePath = '../datasets/Potsdam/Potsdam_tiff/4_Ortho_RGBIR/top_potsdam_7_7_RGBIR.tif'
    
# trainDataset = SingleSceneDataLoader(imagePath, args.trainingPatchSize, args.trainingStrideSize, args.dataset)
trainDataset = PatchLoader(patch_path, test_scene_ids, 448, args.dataset, args.no_patches)

trainLoader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size = args.trainingBatchSize, shuffle = True, num_workers=1) 


print('Training', args.dataset + '_Model.pth')
for epochIter in range(4):
    for batchStep, batchData in enumerate(tqdm(trainLoader)):
                    
        view1 = batchData['I1'].float().cuda()
        view2 = batchData['I2'].float().cuda()
        # randomShufflingIndices = torch.randperm(view2.shape[0])
        # viewShuffled = view2[randomShufflingIndices, :, :, :]
            
                
        for trainingInsideIter in range(1):
            optimizer.zero_grad()

            rot_angle = random.randint(1, 3) # random rotation from 90, 180, 270
            view2 = torch.rot90(view2, rot_angle, [2, 3]) # augmented and rotated image

            projection1, projection2 = model(view1, view2)
            # _, projection2Shuffled = model(view1, viewShuffled)
            
            projection2 = torch.rot90(projection2, -rot_angle, [2, 3]) # rotate the features back
            
            _, prediction1 = torch.max(projection1, 1)
            _, prediction2 = torch.max(projection2, 1)
            # _, prediction2Shuffled = torch.max(projection2Shuffled, 1)
            
            lossPrimary1 = lossFunction(projection1, prediction1) 
            lossPrimary2 = lossFunction(projection2, prediction2)
            
            lossSecondary1 = lossFunctionSecondary(projection1, projection2)
            # lossSecondary2 = -lossFunctionSecondary(projection1, projection2Shuffled)
            
            lossPrimary = (lossPrimary1 + lossPrimary2)/2
            lossTotal = (lossPrimary1 + lossPrimary2 + lossSecondary1)/3
            # lossTotal = (lossPrimary1 + lossPrimary2 + lossSecondary1 + lossSecondary2)/4

            if epochIter <= 1:
                lossPrimary.backward()
            else:
                lossTotal.backward()
            # lossTotal.backward()
            optimizer.step()

torch.save(model, saveModelPath)