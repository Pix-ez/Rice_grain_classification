#!/bin/bash

# Download the dataset
curl -L -o rice-image-dataset.zip \
https://www.kaggle.com/api/v1/datasets/download/muratkokludataset/rice-image-dataset

# Unzip the dataset
unzip rice-image-dataset.zip -d rice-image-dataset
