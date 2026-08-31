# Robust Detection of AI‑Generated Images Under Real‑World Transformations
## Project Overview
This project implements a deep learning‑based detector that distinguishes AI‑generated images from authentic photographs, 
with a focus on robustness to common real‑world transformations such as JPEG compression, blur, resizing, noise, color adjustments, cropping and rotation.
The solution uses a lightweight convolutional neural network (ConvNeXt_Small) fused with radially averaged spectral cues, fine‑tuned on public datasets (saberzl/SID_Set, OwensLab/CommunityForensics), together with an augmentation pipeline during training to simulate the post‑processing artifacts often encountered when images are shared on social media or messaging platforms. Furthermore, methods from Spectral Tail Auxiliary Learning (arXiv:2605.22751v1) and Any-Resolution AI-Generated Image Detection by Spectral Learning (arXiv:2411.19417) are explored.
## Results
The following shows the test results on the original, moderately transformed and severely transformed (a random combination of the common real-world transformations mentioned in the Project Overview including rotations are multiples of 90 degrees) versions of the combined dataset of ``COCOval2017`` and ``DALL·E Advanced``.

![results](results/logo.png)

It turns out that Spectral Tail Auxiliary Learning is unlikely to help for our training dataset since the tail uplift does not actually exist for our dataset of AI-generated images and cannot differentiate AI-generated images from real images, as shown in ``archive_STAL``.

Meanwhile, a preliminary experiment shows that the spectral learning method (that investigates conditional distribution of the spectra) implemented by SPAI: Spectral AI-Generated Image Detector is unlikely to improve the ROC for our dataset, as shown in ``masked_frequency_experiment``.
## Setup
Download the latest release and ensure Python3 and related packages and modules are available.
## Reproducing the Results
Unzip the release to get a folder ``detector``. Open terminal in the folder and in the command line, run

`python detect.py --image_dir /path/to/the/image/folder --output_json /output/path --model_path /path/to/weights`

where ``--image_dir`` is compulsory, while ``--output_json`` is by default ``predictions.json`` and ``--model_path`` by default ``weights.pth``

A JSON file will be generated, containing ``image_path`` and ``pred`` for each image, where ``pred`` is the confidence score that the image is AI-generated.
## Robustness Evaluation Summary

## Error Analysis Note

## Limitations and Future Enhancements
The size and diversity of the datasets used are limited by our local storage.
## Contributions

