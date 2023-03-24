# Image colorization methods: a Review

The colorization of greyscale images is an ill-posed
problem that was approached in different ways in literature.
This project provides a comparative analysis concerning
five pre-trained colorization models and a cartoonization-based
baseline of our invention. The performances are assessed
through both quantitative and qualitative metrics,
with a final evaluation of the results with respect to image
filtering.

# Dataset

We considered three types of images: 4023 originally
colored images from five different datasets, 18 originally
black and white images from various artists and 180 filtered
images obtained starting
from 18 originally colored images.
Our data includes heterogeneous images, representing
different environments, situations and subjects, coming
from various sources.

<img align="left" width="60%" src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9886d32f12de1853a68ea5309520165e9d5aaf03/report/datasets.png">

<br />
<br />
<br />

- Imagenette: https://github.com/fastai/imagenette
- Pascal: https://deepai.org/dataset/pascal-voc
- Places: https://paperswithcode.com/dataset/places205
- Birds: https://www.kaggle.com/gpiosenka/100-bird-species
- Flowers: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

<br />
<br />
<br />
<br />

# Colorization models overview

<p align="center">
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/dahl.png"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/ECCV16.png"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/Siggraph17.png"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/InstColorization1.png"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/InstColorization2.png"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/InstColorization3.png"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/ChromaGAN1.png"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/ChromaGAN2.png"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/ChromaGAN3.png"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/ChromaGAN4.png"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/ChromaGAN5.png"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/ChromaGAN6.png"/>
</p>


Here you can find the pretrained models considered in this project:
- Dahl: https://tinyclouds.org/colorize/ (Download section)
- Zhang eccv_16: https://github.com/richzhang/colorization/blob/master/colorizers/eccv16.py - related paper [here](https://arxiv.org/abs/1603.08511)
- Zhang siggraph17: https://github.com/richzhang/colorization/blob/master/colorizers/siggraph17.py - related paper [here](https://arxiv.org/abs/1705.02999)
- ChromaGAN: https://github.com/pvitoria/ChromaGAN - related paper [here](https://arxiv.org/abs/1907.09837)
- InstColorization: https://github.com/ericsujw/InstColorization - related paper [here](https://arxiv.org/abs/2005.10825)

### Requirements
- python 3.6 or 3.8
- virtualenv wrapper: https://virtualenvwrapper.readthedocs.io/en/latest/

### Dahl
- download the pretrained model and place the `colorize.tfmodel` file in the `pre-trained-models` folder
- create a virtual environment: `mkvirtualenv --python=python3 dahl`
- install the requirements: `pip install -r requirements_dahl.txt`
- position yourself into the following folder: `cd src/models`
- run the model: `python3 dahl.py`
    
### Eccv16 and Siggraph17
- create a virtual environment: `mkvirtualenv --python=python3 zhang`
- install the requirements: `pip install -r requirements_zhang.txt`
- position yourself into the following folder: `cd src/models`
- run the model: `python3 Eccv16andSiggraph17.py`

### ChromaGAN
- download the pretrained model and place the `ChromaGAN.h5` file in the `pre-trained-models` folder
- create a virtual environment using python 3.6: `mkvirtualenv --python=python3 chromaGAN`
- install the requirements: `pip install -r requirements_chromaGAN.txt`
- position yourself into the following folder: `cd src/models`
- run the model: `python3 chromaGAN.py`

### InstColorization
The code to run this model is contained in the notebook: `src/models/InstColorization.ipynb`.

### Repository Guideline
- `img` folder
    - `img/colorized` folder: colorized images by our models
    - `img/filtered` folder: filtered images
    - `img/original` folder: original images
- `pre-trained-models` folder: saved pre-trained models (it should also contain ChromaGAN.h5 and colorize.tfmodel)
- `report` folder: report PDF and some slides
- `requirements` folder: requirements for each model
- `resources` folder:
    - `resources/classification` folder: text files containing the performance of AlexNet (pre-trained on ImageNET) when classifying the colorized images. We also performed fine-tuning of AlexNET on Birds and Flowers datasets and feature extraction of AlexNET on a subset of ImageNet
    - `resources/img_classes` folder: info about some datasets labels
    - `resources/LPIPS` folder: text files containing the LPIP metric results
    - `resources/PSNRandSSIM` folder: text files containing the PSNR and SSIM metrics results
- `src` folder: python code for AlexNET class definition, dataset normalization, image filtering, image cartoonization, classification, fine-tuning, feature extraction, metrics computation, graphical representation of the Turing Test results 
    - `src/models` folder: python code for running the colorization models


