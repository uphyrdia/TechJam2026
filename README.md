# Robust Detection of AI‑Generated Images Under Real‑World Transformations
## Project Overview
This project implements a deep learning‑based detector that distinguishes AI‑generated images from authentic photographs, 
with a focus on robustness to common real‑world transformations such as JPEG compression, blur, resizing, noise, color adjustments, and cropping.
The solution uses a lightweight convolutional neural network (ConvNeXt_Small) fine‑tuned on public datasets (saberzl/SID_Set, OwensLab/CommunityForensics),
and attempts to exploit Spectral Tail Auxiliary Learning (arXiv:2605.22751v1), together with an augmentation pipeline during training to simulate 
the post‑processing artifacts often encountered when images are shared on social media or messaging platforms.
## Setup
Download ``detect_aigc.py``, ``model.py`` and ``aigc_detector.pth`` into the same folder and ensure Python3 is available.
## Reproducing the Results
In command line, run

`python3 detect_aigc.py --image_dir /path/to/the/image/folder`

A JSON file will be generated, containing ``image_path`` and ``pred`` for each image, where ``pred`` is the confidence score that the image is AI-generated.
## Updates
It turns out that Spectral Tail Auxiliary Learning (arXiv:2605.22751v1) is unlikely to help for our training dataset since the tail uplift cannot actually differentiate AI-generated images from real images, as shown in ``tail_uplift_histograms_comminutyforensics.png`` and ``tail_uplift_histograms_SID.png`` produced by ``compare_tail_uplift_distributions.ipynb``.
## Limitations and Future Enhancements
## Contributions