# IDP2021-HSI-Segmentation-by-HRNet-and-ResNet

Coursework project: Industrial Project - [UEF, Finland](https://www.uef.fi/en/unit/school-of-computing)

- Project owners: [Thong Nguyen](https://github.com/ThongNguyen551), [Borhan Sumon](https://github.com/Borhan-Uddin), [Agha Danish](https://github.com/AghaDanish98), [Gemal](https://github.com/JemalHamid)
- This project aims to implement segmentations algorithm for Medical Hyperspectal Imaging (HSI), then implement the GUI for visualization. The DL methods used are HRNet and ResNet.
 
## Dataset
- Place your dataset as follow:

```
dataset/annotations/
dataset/images/
dataset/masks/
```

## Dataset
- You can run with our model by placing it here:

```
libs/model/
```

## HRNet
### Prerequisites
1. Install requirements ``` pip install -r requirement.txt ```
2. ``` pip install tensorflow-gpu==2.4.1 ``` (For running with GPU)
3. Change the model configuration depends on your needs in ```libs/config.py```


### Training
```
python tools/train.py train
```
### Evaluation
```
python tools/evaluation.py 
```

### Open the GUI
```
python ui/main_ui.py 
```

## ResNet
First, you should go to the folder ResNet and follow the instruction below

### Prerequisites
1. Install requirements ``` pip install -r resnet/requirement.txt ```


### Training
```
python resnet/training.py
```
### Evaluation
```
python resnet/evaluate.py 
```
