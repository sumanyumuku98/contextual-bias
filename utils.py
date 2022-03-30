import os
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO

# Coco Class Names
class_names = [
                   'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                   'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                   'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']


# Coefficient of variation for protected class within bias-classes for selected Ids.
def imbalance_ratio(cocoObj, imageIdsUsed:list, inspectClass:str, biasClasses:list):
    """
    Args:
        cocoObj: Coco obj generated from pycocotools and ann_path
        imageIdsUsed: List of integer Ids of the images selected
        inspectClass: Bias class under inspection. In our case "cup"
        biasClasses: List of string containing top-k bias classes.
    """

    classes=[]
    for cat in biasClasses:
        classes.append(cat)
    countDict={}
    for cat in biasClasses:
        countDict[cat]=0

    for cat in biasClasses:
        temp_count=0
        tempcatIds = cocoObj.getCatIds([inspectClass,cat])
        imgIds_cats = cocoObj.getImgIds(catIds = tempcatIds)
        for idx in imageIdsUsed:
            if idx in imgIds_cats:
                temp_count+=1
        countDict[cat] = temp_count

    for k, v in countDict.items():
        countDict[k] = float(v/len(imageIdsUsed))

    array = np.array(list(countDict.values()), dtype=np.float32)
    mean = np.mean(array, axis=-1)
    stddev = np.std(array, axis=-1)
    return stddev/mean

# Count dictionary of protected attribute with co-occuring classes for the selected images.
def genCocoStats(cocoObj, imageIdsUsed:list, inspectClass:str, biasClasses:list, normalize=False):
    """
    Args:
        cocoObj: Coco obj generated from pycocotools and ann_path
        imageIdsUsed: List of integer Ids of the images selected
        inspectClass: Bias class under inspection. In our case "cup"
        biasClasses: List of string containing top-k bias classes.
        save_path: Path to save the result.
    """
    classes = [inspectClass]
    for cat in biasClasses:
        classes.append(cat)
    countDict={}
    for cat in biasClasses:
        countDict[cat]=0

    for cat in biasClasses:
        temp_count=0
        tempcatIds = cocoObj.getCatIds([inspectClass,cat])
        imgIds_cats = cocoObj.getImgIds(catIds = tempcatIds)
        for idx in imageIdsUsed:
            if idx in imgIds_cats:
                temp_count+=1
        countDict[cat] = temp_count
    
    if normalize:
        for k,v in countDict.items():
            countDict[k] = np.round(v/len(imageIdsUsed),3)

    sorted_dict  = {k:v for k,v in sorted(countDict.items(), key=lambda item: item[1], reverse=True)}


    return sorted_dict


# Get co-occuring class names for COCO for bias_class
def get_class_names(ann_path, bias_class):
    coco = COCO(ann_path)
    count_dict={}
    cats = coco.loadCats(coco.getCatIds())
    for cat in cats:
        catids = coco.getCatIds([bias_class,cat["name"]])
        imgids = coco.getImgIds(catIds=catids)
        count_dict[cat["name"]] = len(imgids)
    sorted_dict  = {k:v for k,v in sorted(count_dict.items(), key=lambda item: item[1], reverse=True)}
    return sorted_dict

def ir_numpy(array, epsilon=1e-4):
    return np.std(array, axis=-1)/(np.mean(array, axis=-1)+epsilon)

def k_center_fair(labelArr:np.ndarray, imageIds:list, k:int):
    """
    Args:
        points_: Unlabelled Pool
        k: number of points to select
    returns:
        selected: Selected Pool of points
    """
    budget=k
    selected_ids = []
    assert len(labelArr)==len(imageIds)
    ind1 = np.random.choice(len(imageIds))
    # while not labelArr[ind1].all():
    #     ind1=np.random.choice(len(imageIds))
        # print(ind1)

    selected_ids.append(ind1)
    k-=1
    pbar = tqdm(total=budget-1)
    while k!=0:
        unselected_ids=list(set(range(len(imageIds))) - set(selected_ids))
        unselected_ids=[id_ for id_ in unselected_ids if labelArr[id_].any()]
        
        selected_labels = np.sum(labelArr[selected_ids], axis=0)
        unselected_labels = labelArr[unselected_ids]
        selected_label_repeated=np.tile(selected_labels, (unselected_labels.shape[0], 1))
        new_label_arr = selected_label_repeated+unselected_labels
        ir_list=np.apply_along_axis(ir_numpy, -1, new_label_arr)        
        
        min_ind = np.argmin(ir_list)
        p3 = unselected_ids[min_ind]
        selected_ids.append(p3)
        pbar.update(1)
        k-=1
    
    pbar.close()
    selected_img_ids = [imageIds[id_] for id_ in selected_ids]

    assert len(set(selected_img_ids))==budget

    return selected_img_ids


def classPresent(coco_obj, img_id:int, class_names:list):
    
    coco = coco_obj
    class_present=[]
    # catName_to_id={}
    id_to_catName={}
    catids = coco.getCatIds(class_names)

    for cat in class_names:
        # catName_to_id[cat] = coco.getCatIds(cat)[-1]
        id_to_catName[coco.getCatIds(cat)[-1]] = cat

    annids = coco.getAnnIds(imgIds=img_id, catIds=catids, iscrowd=None)
    anns = coco.loadAnns(annids)

    for item in anns:
        class_present.append(id_to_catName[item["category_id"]])

    return list(set(class_present))


def genGT_vector(coco_obj, image_id, class_list):
    arr = np.zeros(len(class_list))
    class_present = classPresent(coco_obj, image_id, class_list)
    for item in class_present:
        ind = class_list.index(item)
        arr[ind] = 1.0
    return arr


def genPointMatrix(coco_obj, image_id_list, class_list):
    myList=[]
    for image_id in image_id_list:
        myList.append(genGT_vector(coco_obj, image_id, class_list))
    return np.stack(myList)