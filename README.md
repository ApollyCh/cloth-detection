# Cloth Detection and Segmentation

## Project Idea

The aim of this project is to develop a computer vision system capable of
detecting and segmenting various types of clothing items in images. The system
will focus on two main tasks:

1. **Clothing Classification**: Identifying and categorizing of clothing types
   (e.g., shirts, pants, dresses).
2. **Clothing Segmentation**: Accurately separating clothing items from the
   image background for enhanced detection and identification in different
   environments.

## Technique/Method

We plan to implement a deep learning-based approach, utilizing state-of-the-art
models for both the classification and segmentation tasks.

### Classification

For clothing classification, we will fine-tune a pre-trained convolutional
neural network (CNN) model. This will enable efficient feature extraction and
recognition of clothing categories in images.

### Segmentation

For clothing segmentation, we will employ advanced segmentation models like
**U-Net** [3] or **DeepLab** [4].

## Dataset Explanation and Link

[DeepFashion Dataset](https://github.com/switchablenorms/DeepFashion2) [1] and
[ModaNet Dataset](https://github.com/eBay/modanet) [2]

## Timeline and Individual Tasks

| Week       | Task                                              | Responsible Member     |
| ---------- | ------------------------------------------------- | ---------------------- |
| Week 9     | Data exploration, preprocessing, and augmentation | Apollinaria Chernikova |
| Week 10    | Fine-tuning the classification model              | Apollinaria Chernikova |
| Week 11    | Training the segmentation model                   | Egor Machnev           |
| Week 12-13 | Testing, debugging, and performance evaluation    | All members            |

## References

[1] Liu _et al_. "DeepFashion: Powering robust clothes recognition and
retrieval," 2016. [Link](https://ieeexplore.ieee.org/document/7780493)

[2] Zheng _et al_. "ModaNet: A large-scale street fashion dataset with polygon
annotations," 2018. [Link](https://arxiv.org/abs/1807.01394)

[3] Ronneberger _et al_. "U-Net: Convolutional Networks for Biomedical Image
Segmentation," 2015. [Link](https://arxiv.org/abs/1505.04597)

[4] Chen _et al_. "Rethinking Atrous Convolution for Semantic Image
Segmentation," 2017. [Link](https://arxiv.org/abs/1706.05587)
