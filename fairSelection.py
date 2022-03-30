import sys
import os
import numpy as np
import json
import argparse
from pycocotools.coco import COCO
from utils import class_names, get_class_names, genCocoStats, ir_numpy, k_center_fair, genPointMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Selecting contextually fair data for a protected attribute with its co-occuring classes")
parser.add_argument("--ann_dir", type=str, default="./data")
parser.add_argument("--data_mode", type=str, choices=["train", "val"], default="train")
parser.add_argument("--bias_class", type=str, choices=class_names+["man", "woman"], required=True)
parser.add_argument("--save_path", type=str, default="./out_files/")
parser.add_argument("--seed", type=int, default=23)
parser.add_argument("--budget", type=int, required=True)
parser.add_argument("--cooccur", type=str, choices=["topK", "bottomK", "custom"], default="topk")
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--custom_list", type=str, default="./data/custom_cooccur.txt")

args=parser.parse_args()
np.random.seed(args.seed)

dmode=None
ann_path=None
gender_path=None

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.data_mode=="train":
    dmode="train2017"
else:
    dmode="val2017"

ann_path= os.path.join(args.ann_dir, "annotations/instances_%s.json" % dmode)

if args.bias_class=="man":
    gender_path= os.path.join(args.ann_dir, "bias_splits/man_%s.txt" % args.data_mode)
elif args.bias_class=="woman":
    gender_path= os.path.join(args.ann_dir, "bias_splits/woman_%s.txt" % args.data_mode)


bias_class_list=None
total_ids=None
if args.cooccur=="custom":
    with open(args.custom_list, "r") as f:
        bias_class_list=f.readlines()
    bias_class_list=[item.strip() for item in bias_class_list]
    if (args.bias_class in ["man", "woman"] and "person" in bias_class_list) or (args.bias_class=="person" and ("man" in bias_class_list or "woman" in bias_class_list)):
        print("Incorrect Config")
        sys.exit(-1)
else:
    sorted_dict=None
    if args.bias_class in ["man", "woman"]:
        with open(gender_path, "r") as f:
            gender_ids=f.readlines()
        gender_ids=[int(x) for x in gender_ids]
        cocoObj=COCO(ann_path)
        sorted_dict=genCocoStats(cocoObj, gender_ids, "person", class_names[1:])
    else:
        sorted_dict=get_class_names(ann_path, args.bias_class)

    if args.cooccur=="topk":
        bias_class_list=list(sorted_dict.keys())[1:args.k+1]
    else:
        bias_class_list=list(sorted_dict.keys())[-args.k:]

print("Contextually Biased Classes:")
print(bias_class_list)

cocoObj=COCO(ann_path)
completeDatasetIds=[]
if args.bias_class in ["man", "woman"]:
    with open(gender_path, "r") as f:
        gender_ids=f.readlines()
    gender_ids=[int(x) for x in gender_ids]
    allcocoIds=[]
    for cat in bias_class_list:
        catIds=cocoObj.getCatIds([cat])[-1]
        imgIds_coco=cocoObj.getImgIds(catIds=catIds)
        allcocoIds+=imgIds_coco
    allcocoIds=list(set(allcocoIds))
    completeDatasetIds=list(set(gender_ids).intersection(set(allcocoIds)))
else:
    for cat in bias_class_list:
        catIds=cocoObj.getCatIds([args.bias_class, cat])
        imgIds_coco=cocoObj.getImgIds(catIds=catIds)
        completeDatasetIds+=imgIds_coco
    completeDatasetIds=list(set(completeDatasetIds))


print("Length of the data pool: %d" % len(completeDatasetIds))
selected_ids=None
if args.budget>= len(completeDatasetIds):
    selected_ids=completeDatasetIds
else:
    dataMatrix=genPointMatrix(cocoObj, completeDatasetIds, bias_class_list)
    selected_ids=k_center_fair(dataMatrix, completeDatasetIds, args.budget)

selected_data_matrix=genPointMatrix(cocoObj, selected_ids, bias_class_list)
normalized_arr=np.sum(selected_data_matrix, axis=0)/np.sum(selected_data_matrix)
# print(normalized_arr)

print("Coefficient of Variation of selected pool: %.3f" % ir_numpy(normalized_arr))

with open(os.path.join(args.save_path, "selected_ids.txt"), "w") as f:
    for id_ in selected_ids:
        f.write("%d\n" % id_)




