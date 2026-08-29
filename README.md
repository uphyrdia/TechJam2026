# Robust Detection of AI‑Generated Images Under Real‑World Transformations
## Project Overview
This project implements a deep learning‑based detector that distinguishes AI‑generated images from authentic photographs, 
with a focus on robustness to common real‑world transformations such as JPEG compression, blur, resizing, noise, color adjustments, and cropping.
The solution uses a lightweight convolutional neural network (ConvNeXt_Small) fine‑tuned on public datasets (saberzl/SID_Set, OwensLab/CommunityForensics),
and attempts to exploit Spectral Tail Auxiliary Learning (arXiv:2605.22751v1), together with an augmentation pipeline during training to simulate 
the post‑processing artifacts often encountered when images are shared on social media or messaging platforms.
## Setup
Download detect_aigc.py and aigc_detector.pth into the same folder and ensure Python3 is available.
## Reproducing the Results
In command line, run
``python3 detect_aigc.py --image_dir /path/to/the/image/folder``
A JSON file will be generated, storing the the path to each image and the confidence score that the image is AI-generated.
## Limitations and Future Enhancements
## Contributions