# Robust Detection of AI‑Generated Images Under Real‑World Transformations
## Project Overview
This project implements a deep learning‑based detector that distinguishes AI‑generated images from authentic photographs, 
with a focus on robustness to common real‑world transformations such as JPEG compression, blur, resizing, noise, color adjustments, cropping and rotation.
The solution uses a lightweight convolutional neural network (ConvNeXt_Small) fused with radially averaged spectral cues, fine‑tuned on public datasets (saberzl/SID_Set, OwensLab/CommunityForensics), together with an augmentation pipeline during training to simulate the post‑processing artifacts often encountered when images are shared on social media or messaging platforms. Furthermore, methods from Spectral Tail Auxiliary Learning (arXiv:2605.22751v1) and Any-Resolution AI-Generated Image Detection by Spectral Learning (arXiv:2411.19417) are explored.
## Results and Robustness Evaluation Summary
The zip files ``demo_data/clean_demo.7z``, ``demo_data/transformed_demo.7z``, ``demo_data/severely_transformed_demo.7z`` contain respectively the original, moderately transformed and severely transformed (a random combination of the common real-world transformations mentioned in the Project Overview including rotations of multiples of 90 degrees) versions of the combined dataset of ``COCOval2017`` and ``DALL·E Advanced``.

The following shows the test results on them

![roc](roc_curve_comparison.png)
![histo1](confidence_distribution_1_original.png)
![histo2](confidence_distribution_2_transformed.png)
![histo3](confidence_distribution_3_severely_transformed.png)

In summary, real-world transformations do affect the accuracy of our model (more AI generated images are misclassified as the severity of the transformations increases). However, it is unlikely that images on social media will be ``severely_transformed`` as we did for the original dataset, and the decrease in the AUC of ROC is still within an acceptable range given the small training dataset and a relatively lightweight model.

Alternatives like a spatial-only CNN model or one fused with full 3-channel FFT with a four-layer transformer are also implemented, as in the folders ``spatial_only`` and ``rgb_fft_with_transformer``, but neither of them outperform fusing with radially averaged spectral cues, which is chosen not only because of better AUC and accuracy, but also because its inherent rotational robustness (since Fourier transform commutes with rotation).

On the other hand, it turns out that Spectral Tail Auxiliary Learning is unlikely to help for our training dataset since the tail uplift does not actually exist for our dataset of AI-generated images and cannot differentiate AI-generated images from real images, as shown in ``archive_STAL``.

Meanwhile, a preliminary experiment shows that the spectral learning method (that investigates conditional distribution of the spectra) implemented by SPAI: Spectral AI-Generated Image Detector is unlikely to improve the ROC for our dataset, as shown in ``masked_frequency_experiment``.
## Setup
Download the latest release and ensure Python3 and related packages and modules are installed.
## Reproducing the Results
Unzip the release into a folder. Open terminal in the folder and run

`python detect.py --image_dir demo_original`

A JSON file called ``predictions.json`` will be generated, containing ``image_path`` and ``pred`` for each image, where ``pred`` is the confidence score that the image is AI-generated.

In general, the format is

`python detect.py --image_dir /path/to/the/image/folder --output_json /output/path --model_path /path/to/weights`

where ``--image_dir`` is compulsory and selects the directory of images to be analyzed, while ``--output_json`` is by default ``predictions.json`` and ``--model_path`` is by default ``weights.pth`` in the current working directory.
## Error Analysis Note

## Limitations and Future Enhancements
The size and diversity of the datasets used are limited by our local storage.
## Contributions

