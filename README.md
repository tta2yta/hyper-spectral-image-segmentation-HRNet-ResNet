# hyper-spectral-image-segmentation-HRNet-ResNet

Coursework project: Industrial Project - [UEF, Finland](https://www.uef.fi/en/unit/school-of-computing)

Spectral imaging with help of digital image processing techniques such as Image segmentation has been playing a significant role in solving real world complex problems in numerous fields. Specifically in medical imaging, the visualization plays an important role while diagnosing. Spectral imaging allows us to analyze some extremely small yet very important tissues present in human body while image segmentation algorithms helps us to efficiently classify those spectral information into useful information. In this work, we aim to implement two state-of-the-art Deep Learning methods ResNet and HRNet to do semantic segmentation on given spectral dataset of human placenta tissue. Additionally, we train the model with two main options: full 38 bands of spectral images and 3 bands of spectral images after applying PCA. By looking at our qualitative results, we can say that our models have outperformed the previous groups’ models.
 
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
