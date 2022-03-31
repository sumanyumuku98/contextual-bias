# Does Data Repair Lead to Fair Models? Curating Contextually Fair Data To Reduce Model Bias (WACV 2022)
This is the code accompanying the paper [Does Data Repair Lead to Fair Models?
Curating Contextually Fair Data To Reduce Model Bias](https://openaccess.thecvf.com/content/WACV2022/papers/Agarwal_Does_Data_Repair_Lead_to_Fair_Models_Curating_Contextually_Fair_WACV_2022_paper.pdf) published in IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022.

## Usage
Clone the repo:
```
git clone https://github.com/sumanyumuku98/contextual-bias.git
```
Create Conda env:
```
conda env create -f fair.yml
```
Activate the conda env:
```
conda activate fair.yml
```

## Dataset
Download the [COCO 2017 annotations](https://cocodataset.org/#download) and create a symlink:
```
cd data
ln -s /path to annotations/ .
```
## Visualizing Contextually Biased Categories
Generate a barplot for the protected attribute/class w.r.t other co-occuring classes:
```
python genPlt.py --data_mode train --bias_class man
```
The above example generates co-occuring distribution for `man`. The generated barplots will be saved in `out_figures`.
## Curating Contextually Fair Data
To curate contextually fair data for a protected attribute and fixed budget w.r.t to either `topK`, `bottomK` and `custom` co-occuring categories use:
```
python fairSelection.py --data_mode train --bias_class cup --budget 500 --k 10
```
The above example curates fair data for protected class `cup` with a budget of `500` and co-occuring classes being the top `10`. The selected ids are saved in `out_files`.

## Citation
If you find the work useful, do cite:
```
@InProceedings{Agarwal_2022_WACV,
    author    = {Agarwal, Sharat and Muku, Sumanyu and Anand, Saket and Arora, Chetan},
    title     = {Does Data Repair Lead to Fair Models? Curating Contextually Fair Data To Reduce Model Bias},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {3298-3307}
}

```
