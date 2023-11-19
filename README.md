# DLSR-IQA: Perception and Alignment Assessment Model

## Installation 
```python
pip install numpy
pip install pandas
pip install tensorflow
pip install PIL
pip install scipy
```

## Introduction
This model is used to assess AI-Generated Image(AGI) quality, including perception and text-image alignment.

## Get Started
In `apply_model.py`, correct the path and prompt of image which you want to assess imitating the format of the sample. Then run this file, and prediction score will be printed on the screen.

## Directory Structure
* `alignment_model:` Alignment model on different subsets. Usually using `model_total` or its improvement verison `model_withStair` is enough.
* `perception_model:` Perception model on different subsets. Usually using `model_total` is enough.
* `fetures_numpy:` Image features, used for training.
* `data_processing:` Used to process training data, including extracting image features, text features and capturing part of the image.
* `training:` Python codes for training models.
* `Visualization: ` Used to draw charts for visualization.