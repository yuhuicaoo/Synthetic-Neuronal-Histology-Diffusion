# 🔎 Synthetic Data Generation of Neuronal Histology using Diffusion

## 🎯 Purpose
<p align="justify">
Neuronal histology datasets can be challenging to expand due to the time and resources required to collect and annotate high-quality biological images.I am researching whether diffusion models could be used to generate realistic synthetic neuronal histology data (image & mask), providing an alternative approach to data augmentation for downstream computer vision tasks, like instance segmentation. Supervised by <a href="https://profiles.auckland.ac.nz/h-abbasi">Dr. Hamid Abassi</a> and <a href="https://profiles.auckland.ac.nz/callan-loomes">Callan Loomes</a>.
</p>



## 🧠 Overview
<p align="justify"> 
This project investigates a diffusion-based pipeline for generating synthetic neuronal histology images and corresponding neuronal structures. The models are conditioned on biological factors such as brain region and treatment group, allowing synthetic samples to be generated with specific characteristics. The generated data is evaluated against real histology images to assess its visual and structural similarity and its potential for improving instance segmentation performance.
</p>

## 🔬 Dataset
<p align="justify"> 
The dataset consists of neuronal histology images from a fetal sheep model of hypoxic-ischaemic (HI) injury. Images are categorised according to brain region and treatment group, providing multiple biological conditions for conditional generation. 
</p>



## 🛠️ Tech Stack
| Technology | Purpose |
| --- | --- |
| Python | Programming Language |
| PyTorch | Deep Learning Framework |
| Torchvision | Mask R-CNN & Computer Vision |
| Hugging Face Diffusers | Stable Diffusion 1.5 & ControlNet |
| NumPy | Numerical Computing & Array Processing |
| Matplotlib & Seaborn | Data Visualisation & Plotting |
| Pandas | Data Processing & Analysis |
