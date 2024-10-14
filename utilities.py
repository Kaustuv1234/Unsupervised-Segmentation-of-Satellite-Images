

import numpy as np
import scipy.io as sio
import cv2
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

eps = 1e-14


def generate_mask(image_shape, num_squares, square_size, mask_color=0):
    mask = np.ones(image_shape, dtype=np.uint8) * 255
    for _ in range(num_squares):
        x = np.random.randint(0, image_shape[1] - square_size - 1)
        y = np.random.randint(0, image_shape[0] - square_size - 1)
        mask[y:y+square_size, x:x+square_size] = mask_color

    return mask



def superpixel_correction(arr, sprpxl):
    for i in np.unique(sprpxl):
        segment = np.where(sprpxl == i)
        cls = np.argmax(np.bincount(arr[segment]))
        arr[segment] = cls

        
def max_overlap_matching(cm):
    # rows are gt and cols are outputs
    # Pred_Reassigned = np.copy(prediction) 
    col_sum = cm.sum(axis=0) # add cols
    row_sum = cm.sum(axis=1) # add rows
    mapping = dict()
    for col in range(cm.shape[0]):
        mapping[col] = np.argmax(cm[:, col])
        # Pred_Reassigned[prediction == col] = best_match
    return mapping


def hungarain_matching(cm):   
    # rows are gt and cols are outputs
    gt_ind, pred_ind = linear_sum_assignment(-cm.T)
    return dict(zip(gt_ind, pred_ind))


def max_iou_matching(cm):
    # rows are gt and cols are outputs
    col_sum = cm.sum(axis=0) # add cols
    row_sum = cm.sum(axis=1) # add rows
    mapping = dict()
    for col in range(cm.shape[0]):
        mapping[col] = np.argmax(np.divide(cm[:, col], col_sum[col] + row_sum + 1 - cm[:, col]))
    return mapping



palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter/background (red)
           6 : (0, 0, 0)}       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d
    
def eval_image(gt, pred, num_pixels, true_pos, false_pos, false_neg, cal_classes=6):

    im_row, im_col = np.shape(pred)
    # cal_classes = 5 if noclutter else 6 # no. of classes to calculate scores

    # if noclutter:
    #     gt[gt == 5] = 6 # pixels in clutter are not considered (regarding them as boundary)

    # pred[gt == 6] = 6 # pixels on the boundary are not considered for calculating scores
    num_pixels += im_col * im_row - np.count_nonzero(gt == 6)

    # pred1 = np.reshape(pred, (-1, 1))
    # gt1 = np.reshape(gt, (-1, 1))

    # idx = np.where(gt1==6)[0]
    # pred1 = np.delete(pred1, idx)
    # gt1 = np.delete(gt1, idx)
    
    CM = confusion_matrix(np.reshape(pred, (-1, 1)), np.reshape(gt, (-1, 1)), labels=list(range(cal_classes)))

    for i in range(cal_classes):
        true_pos[i] += CM[i, i]
        false_pos[i] += np.sum(CM[:, i]) - CM[i, i]
        false_neg[i] += np.sum(CM[i, :]) - CM[i, i]

    return num_pixels, true_pos, false_pos, false_neg
    

def divide_into_patches(rows, cols, p, s):
    patch_coords = []          
    x = 0
    while x + p <= cols:
        y = 0
        while y + p <= rows:
            patch_coords.append((x, y, x+p, y+p))
            y += s
        if y != rows-p:
            patch_coords.append((x, rows-p, x+p, rows))
        x += s
    if x != cols-p:
        y = 0
        while y + p <= rows:
            patch_coords.append((cols-p, y, cols, y+p))
            y += s
    patch_coords.append((cols-p, rows-p, cols, rows))
    return patch_coords
'''
# def matchSegmentationResultToOriginalLabel(resultMap, referenceMap):
    
#     ##ADDING 1 to not keep any zero value
#     ##Otherwise zero is an object here (Impervious surfaces)
#     resultMap = resultMap+1
#     referenceMap = referenceMap+1
    
#     ##Finding unique values
#     resultMapUniqueVals = np.unique(resultMap)
    
#     referenceMapUniqueVals,referenceMapUniqueCounts = np.unique(referenceMap, return_counts=True)
#     referenceSortingIndices = np.argsort(-referenceMapUniqueCounts)
#     referenceMapUniqueVals = referenceMapUniqueVals[referenceSortingIndices]
    
    
#     resultToReferenceRelationMatrix = np.zeros((len(resultMapUniqueVals),len(referenceMapUniqueVals)))
    
#     totalIntersection = 0
#     for resultIndex, resultUniqueVal in enumerate(resultMapUniqueVals):
#         resultUniqueValIndicator = np.copy(resultMap)
#         resultUniqueValIndicator[resultUniqueValIndicator!=resultUniqueVal] = 0
#         for referenceIndex,referenceUniqueVal in enumerate(referenceMapUniqueVals):
#             referenceUniqueValIndicator = np.copy(referenceMap)
#             referenceUniqueValIndicator[referenceUniqueValIndicator!=referenceUniqueVal] = 0
#             resultReferenceIntersection = resultUniqueValIndicator*referenceUniqueValIndicator
#             numIntersection = len(np.argwhere(resultReferenceIntersection))
#             totalIntersection = totalIntersection+numIntersection
#             resultToReferenceRelationMatrix[resultIndex,referenceIndex] = numIntersection
            
#     resultMapReassigned = np.zeros(resultMap.shape)
    
#     for referenceIndex,referenceUniqueVal in enumerate(referenceMapUniqueVals):
#         matchesCorrespondingToThisVal = resultToReferenceRelationMatrix[:,referenceIndex]
#         if np.sum(matchesCorrespondingToThisVal)==0: ##this check is important, other python finds a max even in a all-zero column
#             continue
#         maximizingIndex = np.argsort(matchesCorrespondingToThisVal)[-1]
#         resultMapOptimumMatch = resultMapUniqueVals[maximizingIndex]
#         resultMapReassigned[resultMap==resultMapOptimumMatch] = referenceUniqueVal
#         resultToReferenceRelationMatrix[maximizingIndex,:] = 0
        
#     ##Subtracting 1 to keep values as it were
#     resultMapReassigned = resultMapReassigned-1
    
#     ##if some label in the result map has not been assigned to any class of reference, then it will have value -1 at this stage
#     ##we reassign it to 6, since "6" indicates undefined class in Potsdam dataset
#     resultMapReassigned[resultMapReassigned==-1]=6 
    
#     return resultMapReassigned.astype(int)
    '''