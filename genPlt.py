import os
import numpy as np
import json
import argparse
from pycocotools.coco import COCO
from utils import class_names, get_class_names, genCocoStats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

parser = argparse.ArgumentParser(description="Generating Bar-Plot for protected attribute")
parser.add_argument("--ann_dir", type=str, default="./data")
parser.add_argument("--data_mode", type=str, choices=["train", "val"], default="train")
parser.add_argument("--bias_class", type=str, choices=class_names+["man", "woman"], required=True)
parser.add_argument("--save_path", type=str, default="./out_figures/")
args=parser.parse_args()

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

print("Generating a plot...")
if args.bias_class in ["man", "woman"]:
    with open(gender_path, "r") as f:
        gender_ids=f.readlines()
    gender_ids=[int(x) for x in gender_ids]
    cocoObj=COCO(ann_path)
    sorted_dict=genCocoStats(cocoObj, gender_ids, "person", class_names[1:])
else:
    sorted_dict=get_class_names(ann_path, args.bias_class)
    
newDict={"Classes":list(sorted_dict.keys())[1:], "Frequency": list(sorted_dict.values())[1:]}

df=pd.DataFrame.from_dict(newDict)
fig, ax = plt.subplots(figsize=(70, 10))
sns.barplot(data=df, x="Classes", y="Frequency", ax=ax)
ax.set_title("Co-occurence distribution of %s" % args.bias_class)
ax.set_xlabel("Co-occuring Classes")
ax.set_ylabel("Frequency")
fig.savefig(os.path.join(args.save_path, "%s_bar.png" % args.bias_class))

