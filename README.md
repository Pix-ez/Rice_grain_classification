# Rice Grain Image Classification

## Dataset
For this task, I selected a recommended dataset containing 75,000 images of rice grains. The images feature consistent lighting with a black background, isolating the grain without background noise.

**Source:** LInk[https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset]

**Classes:**
The dataset consists of 5 distinct rice grain types:
*   **Arborio:** 0
*   **Basmati:** 1
*   **Ipsala:** 2
*   **Jasmine:** 3
*   **Karacadag:** 4

## Model Architecture & Training
I selected **MobileNetV3 Small** as the base model for transfer learning due to its lightweight architecture, speed, and proven accuracy on ImageNet.

*   **Architecture Modifications:** Replaced the final layer with a custom linear layer mapping to the 5 output classes.
*   **Optimizer:** Adam
*   **Learning Rate:** 0.001
*   **Loss Function:** Cross Entropy Loss
*   **Epochs:** 3

**Training Results:**
*   **Validation Accuracy:** 99.82%
*   The model achieved strong precision, recall, and F1-scores across all classes. (See notebook for detailed graphs).

## Quantization
The trained model was quantized using **Post Training Quantization (PTQ)** via the Torch runtime and ExecuTorch library.
exported model using QAT with XNNPACK backend for android which supports CPU runtime with wide operations. I experimented with different configurations:

1.  **Per-Channel Quantization (Convolution layers) & Per-Tensor (FC layers):**
   
2.  **Per-Tensor Quantization (All layers):**
  
*Note: I was unable to reduce the model size below 1MB while maintaining acceptable accuracy evebn (>60%) using MobileNetV3 as the base.*

## Performance Metrics
*All runs were performed with batch size 1 in Python (not on Android device).*

| Model Variant | Filename | Model Size | Accuracy | Avg Inference Time | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base PyTorch (CUDA)** |` rice_model__epoch_0_loss_0.01.pth` | 5.86 MB | 99.00% | 6.89 ms | 145.23 img/sec |
| **Base PyTorch (CPU)** | `rice_model__epoch_0_loss_0.01.pth` | 5.86 MB | 100.00% | 20.02 ms | 49.96 img/sec |
| **Quantized (Per-Channel)** | `rice_model_ptq_optimized_99.pte` | 4.17 MB | 99.01% | 9.98 ms | 100.18 img/sec |
| **Quantized (Per-Tensor)** | `rice_model_quantized_65_.pte` | 1.66 MB | 65.01% | 37.20 ms | 26.88 img/sec |

## Challenges
1.  **MobileNetV3 Architecture:** The depthwise separable convolution layers possess a large range of values. Standard quantization often breaks these layers, leading to significant drops in accuracy.
2.  **Quantization Aware Training (QAT):** I attempted QAT but could not achieve satisfactory accuracy post-quantization; these results are not included in the final notebook.
3.  **Calibration Samples:** Varying the number of calibration samples for PTQ did not result in significant accuracy differences.

## Usage
To run predictions using the script:

```bash
python predict.py "example_rice_image/Basmati/basmati (4751).jpg"
```
```bash
python predict.py "my_image.jpg" --model "models/rice_model_ptq_optimized_99.pte"
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
where we can select models from list and image of rice grain and test model running directly on android device 
andorid device used here= CPU:Mediatek Dimensity 800U , RAM: 8GB, ANDROID: v14

[![Watch the Demo](https://youtube.com/shorts/bXC0zEWvRd8)](https://youtube.com/shorts/bXC0zEWvRd8)