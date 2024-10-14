import os
import sys
import torch
import torchvision
import numpy as np
import cv2 
import argparse
import tifffile 
from tqdm import tqdm
from skimage.filters.rank import modal
from skimage.morphology import disk
from sklearn.metrics import confusion_matrix
from utilities import * 
import segmentation_models_pytorch as smp
from skimage.segmentation import slic



parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='Vaihingen', choices=['Potsdam_RGB', 'Potsdam_RGBIR', 'Vaihingen'], help='choose dataset') 
parser.add_argument('--classes', type = int, default = 6, help = '3 or 6 classes in ground truth')
parser.add_argument('--nFeaturesFinalLayer', type = int, default = 8, help = 'Number of features in the final classification layer')
parser.add_argument('--modelName', type = str, default = 'Model_BYOL', choices=['Model_BYOL', 'Model6'], help = 'Model name')
parser.add_argument('--superpixel_refine', action='store_true', default=False)
args = parser.parse_args()
print(args)
eps = 1e-14

manualSeed=40
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
np.random.seed(manualSeed)

if args.classes == 6:
    noclutter=True  ##ignore clutter in computing accuracy
    num_classes = 5 if noclutter else 6
else:
    num_classes = 3

num_pixels = 0 # accumulator for all valid pixels (not including label 0 and 6)
true_pos = np.zeros((num_classes, 1)) # accumulator for true positives
false_pos = np.zeros((num_classes, 1)) # accumulator for false positives
false_neg = np.zeros((num_classes, 1)) # accumulator for false negatives
    

TP = torch.tensor([]) # accumulator for true positives
FP = torch.tensor([]) # accumulator for false positives
FN = torch.tensor([]) # accumulator for false negatives
TN = torch.tensor([]) # accumulator for false negatives


dir_path = f'./trainedModels/{args.dataset}_{args.modelName}_{args.nFeaturesFinalLayer}.pth'
loadModelPath = os.path.join(dir_path)
model = torch.load(loadModelPath)
model.cuda()
model.eval()
model.requires_grad=False

if args.dataset == 'Vaihingen':
    prefix_image = '../datasets/Vaihingen/Vaihingen_tiff/img/top_mosaic_09cm_area'
    prefix_label = '../datasets/Vaihingen/Vaihingen_tiff/gt/top_mosaic_09cm_area'
    suffix_image = '.tif'
    suffix_label = '.tif'
    test_area_ids = ['11', '15', '28', '30', '34']
elif args.dataset == 'Potsdam_RGB':
    prefix_image = '../datasets/Potsdam/Potsdam_tiff/2_Ortho_RGB/top_potsdam_'
    prefix_label = '../datasets/Potsdam/Potsdam_tiff/5_Labels_all/top_potsdam_'
    suffix_image = '_RGB.tif'
    suffix_label = '_label.tif'
    test_area_ids = ['6_9', '6_14', '4_13', '2_10', '7_10', '2_11', '7_13']
elif args.dataset == 'Potsdam_RGBIR':
    prefix_image = '../datasets/Potsdam/Potsdam_tiff/4_Ortho_RGBIR/top_potsdam_'
    prefix_label = '../datasets/Potsdam/Potsdam_tiff/5_Labels_all/top_potsdam_'
    suffix_image = '_RGBIR.tif'
    suffix_label = '_label.tif'
    test_area_ids = ['6_9', '6_14', '4_13', '2_10', '7_10', '2_11', '7_13']
else:
    print('wrong dataset')
    sys.exit()


print('Testing', dir_path)
for areaID in tqdm(test_area_ids):
    imagePath = prefix_image + areaID + suffix_image
    labelPath = prefix_label + areaID + suffix_label

    img = tifffile.imread(imagePath)
    ground_truth = convert_from_color(tifffile.imread(labelPath))
    if args.dataset != 'Vaihingen':
        img = cv2.resize(img, (0, 0), fx = 0.448, fy = 0.448)
        ground_truth = cv2.resize(ground_truth, (0, 0), fx = 0.448, fy = 0.448)
        
    img = img / 255
    n_rows, n_cols, _ = img.shape
    img = torch.from_numpy(img).permute(2, 0, 1).float().cuda().unsqueeze(0)
    output = np.zeros((n_rows, n_cols)).astype(np.uint8)

    if args.classes == 3:
        ground_truth[ground_truth == 4] = 0
        ground_truth[ground_truth == 5] = 1
        ground_truth[ground_truth == 3] = 2

    p = 448  # Shape of patch
    overlap = 32  # overlap on all sides of the patch
    patch_coords = divide_into_patches(n_rows, n_cols, p, p) # do not have overlap

    for x1, y1, x2, y2 in patch_coords:
        # introduce overlap
        a = max(0, x1-overlap)
        b = min(x2+overlap, n_cols)
        c = max(0, y1-overlap)
        d = min(y2+overlap, n_rows)

        # take a patch
        patch = img[:, :, c:d, a:b]
        _, _, patch_h, patch_w = patch.shape

        # get output from model
        with torch.no_grad():
            proj_patch, _ = model(patch, patch) 
            _, pred_patch = torch.max(proj_patch, 1) 

        # # resize the output in case there is a maxpool
        # if pred_patch.shape != (1, patch_h, patch_w):
        #     resize_image = torchvision.transforms.Resize((patch_h, patch_w), antialias=None)
        #     pred_patch = resize_image(pred_patch) 

        # remove overlap
        crop_x = 0 if a == 0 else overlap
        crop_y = 0 if c == 0 else overlap
        pred_patch = pred_patch[:, crop_y:crop_y+p, crop_x:crop_x+p]
        
        # convert to numpy and remove some noise
        pred_patch = torch.squeeze(pred_patch).cpu().numpy().astype(np.uint8)
        pred_patch = modal(pred_patch, disk(3))

        output[y1:y2, x1:x2] = pred_patch
            
            
    # super pixel correction
    if args.superpixel_refine:
        if args.dataset == 'Potsdam_RGBIR':
            imagePath = imagePath.replace("4_Ortho_RGBIR", "2_Ortho_RGB")
            imagePath = imagePath.replace("RGBIR", "RGB")
            
        img = tifffile.imread(imagePath)

        if args.dataset != 'Vaihingen':
            img = cv2.resize(img, (0, 0), fx = 0.448, fy = 0.448)
            superpixels = slic(img, n_segments = 10000, compactness = 20)
        else:
            superpixels = slic(img, n_segments = 10000, compactness = 20)

        superpixel_correction(output, superpixels)

    # map the outputs to get the best output
    cm = confusion_matrix(ground_truth.reshape(-1), output.reshape(-1), labels=list(range(args.nFeaturesFinalLayer))) # rows are gt and cols are outputs
    mapping = max_iou_matching(cm)
    optimal_prediction = output.copy()
    for i in mapping:
        optimal_prediction[output == i] = mapping[i]


    # save output
    if args.superpixel_refine:
        path = f'./results/{args.dataset}_{args.classes}_{args.modelName}_spxl'
    else:
        path = f'./results/{args.dataset}_{args.classes}_{args.modelName}'
    if not os.path.exists(path):
        os.mkdir(path)
    cv2.imwrite(os.path.join(path, f'{areaID}.png'), cv2.cvtColor(convert_to_color(optimal_prediction), cv2.COLOR_RGB2BGR)) 
    # cv2.imwrite('./results/' + args.dataset + '/' + areaID + '.png', cv2.cvtColor(convert_to_color(optimal_prediction), cv2.COLOR_RGB2BGR)) 

    # store metrics for this image
    num_pixels, true_pos, false_pos, false_neg = eval_image(ground_truth, optimal_prediction, num_pixels, true_pos, false_pos, false_neg, num_classes)
    optimal_prediction = torch.tensor(optimal_prediction).cuda()
    ground_truth = torch.tensor(ground_truth).cuda()
    tp, fp, fn, tn = smp.metrics.functional.get_stats(optimal_prediction.clone().detach(), ground_truth.clone().detach(), mode='multiclass', num_classes=num_classes)
    TP = torch.cat((TP, tp), 0)
    FP = torch.cat((FP, fp), 0)
    TN = torch.cat((TN, tn), 0)
    FN = torch.cat((FN, fn), 0)
    # print(np.sum(true_pos), np.sum(false_pos), np.sum(false_neg))
    # print(torch.sum(TP), torch.sum(FP), torch.sum(FN))


iou = np.divide(true_pos, (true_pos + false_pos + false_neg + eps))
precision = np.divide(true_pos, (true_pos + false_pos + eps))
recall = np.divide(true_pos, (true_pos + false_neg + eps))
f1 = np.divide(2 * np.multiply(precision, recall), (precision + recall + eps))
# for i in range(num_classes):
#     P = true_pos[i]/(true_pos[i] + false_pos[i] + eps)
#     R = true_pos[i]/(true_pos[i] + false_neg[i] + eps)
#     f1[i] = 2 * (P * R)/(P + R + eps)
#     iou[i] = true_pos[i]/(true_pos[i] + false_pos[i] + false_neg[i] + eps)

print('\n************************************')
print('SSSS METRICS' + dir_path)
print('************************************')
print('mean f1:', np.mean(f1), '\nmean iou:', np.mean(iou), '\nOA:', np.sum(true_pos)/num_pixels)


# iou = torch.div(torch.sum(TP, 0), (torch.sum(TP, 0)+torch.sum(FP, 0)+torch.sum(FN, 0)+eps))  #tp / (tp + fp + fn)
# print('\n************************************')
# print('MY METRICS' + dir_path)
# print('************************************')
# print('class wise IOU:', iou.numpy(), 'mean:',torch.mean(iou).numpy())

# # Sum true positive, false positive, false negative and true negative pixels over all images and all classes and then compute score.
# print('micro iou', smp.metrics.iou_score(TP, FP, FN, TN, reduction="micro")) 

# # Sum true positive, false positive, false negative and true negative pixels over all images for each label, then compute score for each label separately and average labels scores. This does not take label imbalance into account.
# print('macro iou', smp.metrics.iou_score(TP, FP, FN, TN, reduction="macro"))

# # print('f1_score', smp.metrics.f1_score(TP, FP, FN, TN, reduction="macro"))
# print('macro acc', smp.metrics.accuracy(TP, FP, FN, TN, reduction="macro"))
# print('micro acc', smp.metrics.accuracy(TP, FP, FN, TN, reduction="micro"))





print()
print()