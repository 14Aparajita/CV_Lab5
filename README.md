# Shape Context Descriptor for Character Recognition (CUDA Enabled)

## Overview

This project implements the **Shape Context Descriptor** for character recognition, inspired by the classical work of Belongie et al. The implementation includes:

- Shape Context Descriptor (log-polar histogram representation)
- Character recognition using K-Nearest Neighbors (KNN)
- CUDA acceleration using Numba
- Experimental evaluation on MNIST dataset

This project was developed as part of **Computer Vision Lab 5**.

---

## Objective

To implement the Shape Context Descriptor for character recognition and experiment with GPU acceleration using CUDA.

---

## Methodology

### 1. Preprocessing
- Resize image to fixed dimension
- Binary thresholding
- Edge detection using Canny

### 2. Shape Context Descriptor
- Extract edge points
- Compute pairwise distances
- Normalize distances (scale invariance)
- Compute angular bins
- Construct log-polar histogram
- Aggregate descriptor

### 3. Classification
- Feature extraction for all samples
- Train/Test split
- KNN classifier (k=3)
- Accuracy evaluation

### 4. CUDA Acceleration
- GPU kernel for O(n²) distance matrix computation
- Implemented using Numba CUDA
- Hybrid CPU-GPU processing

---

## Mathematical Background

For each point \( p_i \), shape context is defined as a histogram:

\[
h_i(k) = \# \{ p_j \neq p_i : (r_{ij}, \theta_{ij}) \in bin(k) \}
\]

Where:
- \( r_{ij} \) = normalized radial distance
- \( \theta_{ij} \) = relative angle
- Log-polar binning ensures scale and rotation sensitivity

---

## Results

| Model Version | Accuracy |
|--------------|----------|
| CPU Version  | ~85% (subset MNIST) |
| CUDA Version | Similar accuracy, faster distance computation |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/Shape-Context-Character-Recognition.git
cd Shape-Context-Character-Recognition
pip install -r requirements.txt
