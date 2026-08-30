# Robust Detection of AI‑Generated Images Under Real‑World Transformations
## Project Overview
This project implements a deep learning‑based detector that distinguishes AI‑generated images from authentic photographs, 
with a focus on robustness to common real‑world transformations such as JPEG compression, blur, resizing, noise, color adjustments, and cropping.
The solution uses a lightweight convolutional neural network (ConvNeXt_Small) fine‑tuned on public datasets (saberzl/SID_Set, OwensLab/CommunityForensics),
and attempts to exploit Spectral Tail Auxiliary Learning (arXiv:2605.22751v1) and Any-Resolution AI-Generated Image Detection by Spectral Learning ( 	arXiv:2411.19417), together with an augmentation pipeline during training to simulate 
the post‑processing artifacts often encountered when images are shared on social media or messaging platforms.
## Results
It turns out that Spectral Tail Auxiliary Learning is unlikely to help for our training dataset since the tail uplift cannot actually differentiate AI-generated images from real images, as shown in ``tail_uplift_histograms_comminutyforensics.png`` and ``tail_uplift_histograms_SID.png`` produced by ``compare_tail_uplift_distributions.ipynb``.

Also, a preliminary investigation shows that the spectral learning method implemented by SPAI: Spectral AI-Generated Image Detector is unlikely to improve the ROC for our dataset.
## Setup
Download the latest release and ensure Python3 and related packages and modules are available.
## Reproducing the Results
In command line, run

`python3 detect_aigc.py --image_dir /path/to/the/image/folder`

A JSON file will be generated, containing ``image_path`` and ``pred`` for each image, where ``pred`` is the confidence score that the image is AI-generated.

## Limitations and Future Enhancements
## Contributions