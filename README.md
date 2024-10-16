# Cloth Detection and Segmentation

## Project Idea

The aim of this project is to develop a computer vision system capable of
detecting and segmenting various types of clothing items in images. The system
will focus on two main tasks:

1. **Clothing Classification**: Identifying and categorizing different types of
   clothing (e.g., shirts, pants, dresses).
2. **Clothing Segmentation**: Accurately separating clothing items from the
   image background for enhanced detection and identification in different
   environments.

This project holds potential applications in fields such as online retail,
fashion analytics, and augmented reality by providing more accurate
identification of clothing in real-world images.

[GitHub Repository](https://github.com/ApollyCh/cloth-detection)

## Technique/Method

We plan to implement a deep learning-based approach, utilizing state-of-the-art
models for both the classification and segmentation tasks.

### Classification

For clothing classification, we will fine-tune a pre-trained convolutional
neural network (CNN) model such as ResNet or EfficientNet. This will enable
efficient feature extraction and recognition of clothing categories in images.

### Segmentation

For clothing segmentation, we will employ advanced segmentation models like
**U-Net** [3] or **DeepLab** [4]. These models will be adapted and trained on a
dataset of fashion images to enable accurate pixel-wise clothing segmentation
from diverse backgrounds.

The training process will be conducted in a GPU-accelerated environment to
ensure optimal performance and faster training times.

## Dataset Explanation and Link

We will leverage two large-scale, publicly available datasets for training and
evaluation:

### 1. DeepFashion Dataset [1]

This dataset contains over 800,000 images, annotated with various clothing
attributes, categories, and bounding boxes. It will be used for both
classification and segmentation tasks, providing a comprehensive set of examples
for training.

**Link**: [DeepFashion Dataset](https://github.com/switchablenorms/DeepFashion2)

### 2. ModaNet Dataset [2]

ModaNet is a fashion dataset with detailed polygonal annotations for different
clothing items. It is particularly suited for training the segmentation model,
as it provides precise boundaries for various types of clothing.

**Link**: [ModaNet Dataset](https://github.com/eBay/modanet)

## Timeline and Individual Tasks

| Week    | Task                                              | Responsible Member     |
| ------- | ------------------------------------------------- | ---------------------- |
| Week 9  | Data exploration, preprocessing, and augmentation | Apollinaria Chernikova |
| Week 10 | Fine-tuning the classification model              | Apollinaria Chernikova |
| Week 11 | Training the segmentation model                   | Egor Machnev           |
| Week 12 | Testing, debugging, and performance evaluation    | All members            |
| Week 13 | Final adjustments and deployment                  | Egor Machnev           |

## References

[1] Liu _et al_. "DeepFashion: Powering robust clothes recognition and
retrieval," 2016. [Link](https://ieeexplore.ieee.org/document/7780493)

[2] Zheng _et al_. "ModaNet: A large-scale street fashion dataset with polygon
annotations," 2018. [Link](https://arxiv.org/abs/1807.01394)

[3] Ronneberger _et al_. "U-Net: Convolutional Networks for Biomedical Image
Segmentation," 2015. [Link](https://arxiv.org/abs/1505.04597)

[4] Chen _et al_. "Rethinking Atrous Convolution for Semantic Image
Segmentation," 2017. [Link](https://arxiv.org/abs/1706.05587)
