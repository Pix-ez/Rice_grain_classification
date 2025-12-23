# Rice disease Image Classification

## Dataset
For this task, I selected a recommended dataset containing 3829 images of rice plant leaf. The images feature consistent lighting with different types of leafs and its conditions.

**Source:** LInk[]https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset/data

**Classes:**
The dataset consists of 5 distinct rice disease types: {'Bacterial Leaf Blight': 0, 'Brown Spot': 1, 'Healthy Rice Leaf': 2, 'Leaf Blast': 3, 'Leaf scald': 4, 'Sheath Blight': 5}
*   **Bacterial Leaf Blight:** 0
*   **Brown Spot:** 1
*   **Healthy Rice Leaf:** 2
*   **Leaf Blast:** 3
*   **Leaf scald:** 4
*   **Sheath Blight:** 5

## Model Architecture & Training
I selected **MobileNetV3 Small** as the base model for transfer learning due to its lightweight architecture, speed, and proven great performace on edge hardware.

*   **Architecture Modifications:** Replaced the final layer with a custom linear layer mapping to the 5 output classes.
*   **Optimizer:** Adam with Learning rate sheduler ReduceLROnPlateau
*   **Learning Rate:** 0.001
*   **Loss Function:** Cross Entropy Loss
*   **Epochs:** 20

**Training Results:**
*   **Validation Accuracy:** 90.47%
*   The model achieved strong precision, recall, and F1-scores across all classes. (See notebook for detailed graphs).

## Quantization
The trained model was quantized using **Post Training Quantization (PTQ)** and additionaly trained with **Quantize Aware Traing** for improved accuracy via the Torch runtime and ExecuTorch library.
exported model using QAT with XNNPACK backend for android which supports CPU runtime with wide operations. I experimented with different configurations:

   
1.  **Per-Channel Quantization (All layers):**

2.  **Quantize Aware Training on dataset with adding fake quant layers:**   

*Note: I was unable to reduce the model size below 1MB while maintaining acceptable accuracy even (>60%) using MobileNetV3 as the base.*

## Performance Metrics
*All runs were performed with batch size 1 in Python (not on Android device).*

| Model Variant | Filename | Model Size | Accuracy | Avg Inference Time | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base PyTorch (CUDA)** |` model_epoch17_acc0.9047.pth` | 5.86 MB | 90.00% | 6.89 ms | 100.23 img/sec |
| **Quantized (Per-Channel)** | `rice_model_mobilenetv3_quantized55.pte` | 1.7 MB | 55.01% | 18.98 ms | 65.18 img/sec |
| **QAT Model** | `mobilenetv3_qat_acu76.pte` | 1.65 MB | 76.01% | 15.20 ms | 66.88 img/sec |

## Challenges
1.  **MobileNetV3 Architecture:** The depthwise separable convolution layers possess a large range of values. Standard quantization often breaks these layers, leading to significant drops in accuracy.
2.  **Quantization Aware Training (QAT):** I attempted QAT that significantly improved model from only PTQ.


## Usage
To run predictions using the script:

```bash
python predict.py "example_rice_leaf_image\Bacterial Leaf Blight\aug_0_1595.jpg"
```
```bash
python predict.py "my_image.jpg" --model "models\mobilenetv3_qat_acu76.pte"
```


## System Information
Training and evaluation were performed on Google Colab using the free tier.

*   **Platform:** Linux-6.6.105+-x86_64-with-glibc2.35
*   **Processor:** x86_64
*   **Python:** 3.12.12
*   **PyTorch:** 2.9.0+cu126
*   **CPU:** 1 physical core, 2 logical cores
*   **RAM:** 12.67 GB
*   **GPU:** Tesla T4 (CUDA Available)
*   **GPU Memory:** 14.74 GB


## Integrated these models into simple android app 
where we can select models from list and image of rice disease and test model running directly on android device 
andorid device used here= CPU:Mediatek Dimensity 800U , RAM: 8GB, ANDROID: v14
##DEMO RUNNING ON ANDROID DEVICE

https://github.com/user-attachments/assets/c4cdb963-172a-48f6-aea0-c6a7d3b4e018


