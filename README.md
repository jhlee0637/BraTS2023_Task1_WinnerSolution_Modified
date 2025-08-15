## BraTS 2023 - Task 1 Adult Glioma Segmentation Challenge Winner Team's Solution
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)

*This project focuses exclusively on BraTS 2023 Task 1 components extracted from the original multi-task repository.*    

This repository contains a **modified and streamlined implementation** of the **winning solution** for the BraTS 2023 Task 1 Adult Glioma Segmentation Challenge. The original solution achieved state-of-the-art performance in brain tumor segmentation using advanced deep learning techniques and ensemble methods.

> ### Citation
> **Original Repository:** https://github.com/ShadowTwin41/BraTS_2023_2024_solutions    
> **Paper:** https://arxiv.org/html/2402.17317v2   

### How to Run (in progress)
1. go to `./script/`
2. download BraTS2023 Gli Challenge file following `0.getData.ipynb`
3. train 'GliGAN' model through `1_augmentation.ipynb`    
...

### Key Modifications from Original
What's Included    
✅ **Complete BraTS 2023 Task 1 pipeline**  
✅ **Streamlined codebase** focusing on adult glioma segmentation  
✅ **Enhanced documentation** with step-by-step guides  
✅ **Data download automation** with detailed instructions  

What's Excluded    
❌ BraTS 2024 related components  
❌ Multi-task learning modules  
❌ Experimental features not used in final solution      
    ❌ Registry-based Augmentation      
    ❌ Dataset without label generator      

### Winner Team's System
>**GPU**: 6x NVIDIA RTX 6000 (48GB VRAM each) - *Original team specification*    
>**RAM**: 1024GB system memory    
>**CPU**: AMD EPYC 7402 24-Core Processor    

### Methodology Overview
Stage 1: Data Augmentation Pipeline
1. **Registry-based Augmentation**: Systematic data transformation and registration (this process is excluded)
2. **GliGAN**: Generative Adversarial Network for synthetic brain tumor data generation

Stage 2: Ensemble Segmentation
1. **nnU-Net**: Self-configuring deep learning framework
2. **Swin UNETR**: Vision Transformer-based segmentation model  
3. **BraTS 2021 Winner**: Integration of previous championship solution

### Core Technologies
- **Deep Learning**: PyTorch, MONAI, nnU-Net
- **Medical Imaging**: SimpleITK, NiBabel, DICOM processing
- **Data Augmentation**: Custom GAN implementation (GliGAN)
- **Ensemble Methods**: Multi-model prediction fusion
- **Containerization**: Docker for reproducible deployment