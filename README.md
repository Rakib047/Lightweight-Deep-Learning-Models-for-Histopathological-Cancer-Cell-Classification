# Lightweight Deep Learning Models for Histopathological Cancer Cell Classification

This repository explores lightweight deep learning architectures for the classification of histopathological cancer cell images. By leveraging transfer learning and attention mechanisms, we aim to improve classification accuracy, particularly for small and imbalanced datasets.

## Key Highlights

### Architectures
- **EfficientNet-B0 with SE Blocks**:  
  Incorporates Squeeze-and-Excitation blocks to recalibrate channel features dynamically.

- **EfficientNet-B0 with MSFF**:  
  Enhances multi-scale feature learning for better representation across spatial resolutions.

- **EfficientNet-B0 with CBAM**:  
  Utilizes channel and spatial attention for refined feature learning.

### Modified DeepCMorph Model with Only Classification Module
1. **Residual Block in the Segmentation Module**:
   - **Purpose**: The `ResidualBlock` enhances feature learning by allowing the network to learn identity mappings, mitigating the vanishing gradient problem and promoting efficient learning of complex representations.
   - **Impact**: Improves feature refinement in the segmentation pipeline, which enhances the quality of segmentation and classification maps used by the classification module.

2. **Squeeze-and-Excitation (SE) Block in the Segmentation Module**:
   - **Purpose**: Adds channel-wise attention to focus on the most critical features.
   - **Impact**: Enhances discriminative power of features, improving classification accuracy, especially for challenging classes (e.g., class 2, with a significant recall improvement from 0.25 to 0.92).

3. **Dropout for Regularization**:
   - **Purpose**: Introduces dropout layers to reduce the risk of overfitting by randomly zeroing out a fraction of features during training.
   - **Impact**: Stabilizes the model's performance, especially for small datasets like CRC-VAL-HE-7K (only 7,000 images).

### Datasets
- **[NCT-CRC-VAL-HE-7K](https://zenodo.org/records/1214456)**:  
  A collection of 7,180 colorectal cancer histopathological image patches, classified into 9 tissue types.

- **PathMNIST**:  
  Adapted for training and validation, resized to 3x28x28 dimensions.

### Performance
- Improved metrics across all models:
  - **98% Accuracy**.
  - **0.97 Macro F1-Score** (highlighting balanced performance across all classes).

## Pretrained Models and Base Repository
Our work builds upon the **DeepCMorph** architecture, focusing on improving the classification module.  
For pretrained models and the original implementation, visit:  
[DeepCMorph Repository](https://github.com/aiff22/DeepCMorph)
