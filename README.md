# Robust Detection of AI‑Generated Images Under Real‑World Transformations
## Project Overview
This project implements a deep learning‑based detector that distinguishes AI‑generated images from authentic photographs, 
with a focus on robustness to common real‑world transformations such as JPEG compression, blur, resizing, noise, color adjustments, and cropping.
The solution uses a lightweight convolutional neural network (ConvNeXt_Small) fine‑tuned on public datasets (saberzl/SID_Set, OwensLab/CommunityForensics),
and attempts to exploit Spectral Tail Auxiliary Learning (arXiv:2605.22751v1), together with an augmentation pipeline during training to simulate 
the post‑processing artifacts often encountered when images are shared on social media or messaging platforms.