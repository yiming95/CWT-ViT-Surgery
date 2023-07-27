# CWT-ViT: A Time-Frequency Representation and Vision Transformer-Based Framework for Automated Surgical Skill Assessment


## Requirements

- Python 3.7
- torchviz
- matplotlib
- torchvision>=0.5
- opencv
- pillow
- scipy
- pytorch
- scikit-learn
- pandas
- numpy
- imageio


To install requirements

```
conda env create -f environment.yml
```

## Dataset

Download the JIGSAWS dataset and format into the LOSO/LOUO cross-validation to the directory: ./data.

JIGSAWS dataset is available from the [Link](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/).

To convert CWT images:

```
python cwt_generator.py
```





## Training

To train the model of single branch (left-mtm/right-mtm/psm1/psm2) under LOSO/LOUO, run this command:


```
python train.py --task Suturing --data-path ./data/Processed_dataset/SU_left_mtm_self --weight-path ./su_self_best_weights --epochs 300
```



## Evaluation

To evaluate the model of single branch (left-mtm/right-mtm/psm1/psm2) under LOSO/LOUO, run this command:


```
python predict.py --task Knot_Tying --data-path ./data/Processed_dataset/KT_left_mtm_grs --weight_pth ./kt_grs_best_weights/best_acc.pth
```



## Acknowledgement
- The code is modified based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py. 