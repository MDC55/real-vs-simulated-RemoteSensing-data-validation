# Real vs. Simulated Remote Sensing Data Validation

## Overview

This repository provides a framework for assessing, benchmarking, and validating simulated remote sensing datasets against real-world Earth observation data. The primary objective is to evaluate how accurately synthetic scene generation models (e.g., DIRSIG) reproduce the spectral, spatial, and structural characteristics observed in real sensor data.

The repository includes validation workflows across multiple remote sensing modalities, enabling systematic comparisons between simulated and measured datasets acquired from airborne, UAV, and LiDAR platforms.

---

# Repository Structure

The repository is organized by sensor platform and data modality.

## Simulated vs. Real Airborne Hyperspectral Validation

This module focuses on airborne imaging spectroscopy datasets, including AVIRIS-NG–style hyperspectral imagery.

**Contents**
- Data preprocessing
- Spectral calibration
- Reflectance comparison
- Spectral similarity analysis
- Statistical validation workflows

---

## Simulated vs. Real UAV Validation

This module evaluates high-resolution UAV imagery by comparing simulated and real drone observations.

**Contents**
- Image preprocessing
- Radiometric comparison
- Vegetation index validation
- Statistical validation workflows

---

## Simulated vs. Real LiDAR Validation

This module focuses on the validation of simulated LiDAR data against real-world point cloud measurements.

**Contents**
- Point cloud preprocessing
- Canopy structure comparison
- Height distribution analysis
- Structural metric validation
- Three-dimensional scene evaluation

---

# Objective

The validation framework aims to:

- Compare simulated and real remote sensing datasets across multiple sensor modalities.
- Quantify the spectral, spatial, and structural agreement between synthetic and measured observations.
- Evaluate the realism and reliability of synthetic scene generation models.
- Provide reproducible workflows for remote sensing simulation validation.

---

# Supported Data Types

- Airborne hyperspectral imagery
- UAV multispectral imagery
- LiDAR point clouds

---

# Applications

This repository can be used for:

- Remote sensing simulation validation
- Sensor performance evaluation
- Synthetic dataset benchmarking
- Algorithm testing using simulated and real datasets
- Earth observation research
