# About The Project

Classifying chest X-ray images as normal or not, as well as tuning the hyperparameters of the model, using a TensorFlow/Keras based model.

# Introduction

Deep Learning and Machine Learning have the potential to dramatically improve and speed up the diagnostic of a wide array
of health conditions.

In this analysis, we focus on medical scanner images of healthy and pneumonic patients to train a model able to recognize either case as accurately as possible.

The deep neural network model is built using TensorFLow/Keras, following a CNN (Convolutional neural network) architecture, to optimize the learning.

# Getting Started

## Prerequisites

Python >= 3.8


## 1. Data source

https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

1. Please uncompress the downloaded file
2. Place folders test, train and val under:

    * data/raw_data


## Installation

1. Clone the repo
   ```sh
   cd <directory where to place the repo>
   git clone https://github.com/younessz/tensorflow_keras_x_ray_imaging.git
   ```
2. Setup Python virtual environment
   ```sh
   cd <repo directory>
   # setup Python virtual environment
   python3.8 -m venv xray_env
   source xray_env/bin/activate
   # installing required Python packages
   python3 -m pip install -r requirements.txt
   ```

# References


# Project status

WIP
