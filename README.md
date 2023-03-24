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
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/dahl.png" width="48%"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/ECCV16.png" width="48%"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/Siggraph17.png" width="48%"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/InstColorization1.png" width="48%"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/InstColorization2.png" width="48%"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/InstColorization3.png" width="48%"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/ChromaGAN1.png" width="48%"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/ChromaGAN2.png" width="48%"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/ChromaGAN3.png" width="48%"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/ChromaGAN4.png" width="48%"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/ChromaGAN5.png" width="48%"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/ChromaGAN6.png" width="48%"/>
</p>

Here you can find the pretrained models considered in this project:
- Dahl: https://tinyclouds.org/colorize/ (Download section)
- Zhang eccv_16: https://github.com/richzhang/colorization/blob/master/colorizers/eccv16.py - related paper [here](https://arxiv.org/abs/1603.08511)
- Zhang siggraph17: https://github.com/richzhang/colorization/blob/master/colorizers/siggraph17.py - related paper [here](https://arxiv.org/abs/1705.02999)
- ChromaGAN: https://github.com/pvitoria/ChromaGAN - related paper [here](https://arxiv.org/abs/1907.09837)
- InstColorization: https://github.com/ericsujw/InstColorization - related paper [here](https://arxiv.org/abs/2005.10825)

We propose as baseline a simple autoencoder with mean squared error loss. Here, the encoder should learn a compact representation
of the greyscale input images and the decoder should
generate the corresponding coloured image. 

<p align="center">
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/baseline1.png" width="48%"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/baseline2.png" width="48%"/>
</p>

The model was trained and validated on all the data available to us, but
instead of using the original dataset, we considered the cartoonized version of the images. As we can see from the example, this cartoonization provides a
fine-grained result and exclude noisy elements that could interfere
with the colorization task. This is done in order to produce a more precise and sectorial colorization. 

Therefore, the model produces cartoonized
colored images and from them we take the a and b channels and combine them with the L channel of the original input images. In this way we mantain the original details of the pictures and sometimes we get better results than the ones obtained with the baseline without cartoonization.

The cartoonization model is taken from in [X. Wang and J. Yu - "Learning to cartoonize using white-box cartoon representations" (2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Learning_to_Cartoonize_Using_White-Box_Cartoon_Representations_CVPR_2020_paper.pdf).

# Colorization metrics
The following illustrates the colorization metrics used and the results for the colorization models under consideration.

<p align="center">
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/metrics.png" width="48%"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/metrics_results.png" width="48%"/>
</p>

# Turing test
The Turing test is a qualitative metric based on human
perceptions. Due to our limited resources we just elaborate
a short survey on Google Forms, divided into two sections:
we asked the participants to first evaluate the colorizations
of 3 black and white photographs and then evaluate the
ri-colorizations of 4 originally colored images. The 124 participants had to score how realistic was each
colorization in a scale from 1 (not realistic at all) to 5 (very
realistic). The test includes only the best colorizers, namely
Eccv16, Siggraph17, ChromaGAN and InstColorization.

<p align="center">
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/turing_test.png" width="48%"/>
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/turing_test_results.png" width="48%"/>
</p>

# Image classification with AlexNET

<p align="center">
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/classification_accuracy.png" width="60%"/>
</p>

We consider the AlexNet classifier pre-trained
on ImageNet and we tested it on our subset of ImageNet. The results are reported in the first row of the table above, but since the accuracies are all relatively low, we decided to apply feature
extraction to better focus on our ImageNet subset. In this new setting all the models except the
Baseline are able to outperform the black and white images. 
Therefore we can say that colors play an important role in image classification. 
The best model according to these experiments are ChromaGAN and InstColorization.

For a further comparison, we applied finetuning on AlexNet to perform
classification on the birds and flowers images, which
present more vibrant and diverse colors than our ImageNet
subset. In this new setting we have, a greater gap than before between the original and
the black and white accuracies, meaning that the color is much more relevant in this other dataset to recognize the depicted objects. Indeed, all the models including the
Baseline with cartoonization are able to improve the accuracy
with respect to the black and white images and the best models in this setting are the two from Zhang, that are able to generalize better than the other. Moreover we can notice that the Baseline with cartoonization always reaches a slightly better accuracy than the baseline without colorization.

# Image filtering

<p align="center">
  <img src="https://github.com/silviapoletti/Image-colorization-methods-review/blob/9978d74548e1f96ac6f0b22f16671cf814932555/report/filtering.png" width="60%"/>
</p>

In general, a blurred image is harder to colorize, and the
more blurred the image is, the worse the final colorization
we get. On the contrary,
images with a higher contrast and luminance get a better
colorization by all the models, with just a few exceptions
depending on the specific image.
Clearly, with cartoonization we reach very unrealistic
colors. In the example above, on the left, it looks like the model didn’t recognize
the grass and probably mistook it for water. Probably,
this is not due to the fact that cartoonization discards some
details from the original images, because blurring does the
same. This could be rather due to the fact that the models
are not trained on cartoonized images and expect a completely
different "image style" in input. In conclusion, this is a good example of how the
colorization is an ambiguous task: the model colors the
grass with a plausible green tone, which actually could result
more realistic than the original brown tone.

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


