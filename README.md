# EEG Biometric Authentication System Using CNNs

[![MATLAB](https://img.shields.io/badge/MATLAB-Data_Processing-blue.svg)](#)
[![Machine Learning](https://img.shields.io/badge/Machine_Learning-CNN-orange.svg)](#)
[![Signal Processing](https://img.shields.io/badge/Signal_Processing-EEG-green.svg)](#)

## 📌 Overview
This repository contains the code, data processing pipelines, and presentation materials for my research on **EEG-Based Biometric Authentication Using Convolutional Neural Networks** . This project was conducted under the Dean's Distinguished Fellowship at the University of Wisconsin La Crosse . 

The primary objective was to design and evaluate deep learning models capable of accurately authenticating individuals based on their brainwave patterns, using 14-channel EEG data .

## 📊 Key Results
*   **High Accuracy:** Developed CNN models that achieved **88% subject-level authentication accuracy** across a cohort of 21 participants .
*   **Robust Evaluation:** Validated model performance utilizing rigorous cross-validation techniques and majority voting mechanisms .

## 📂 Repository Contents
To provide a complete view of the research lifecycle, from raw data processing to academic communication, this repository includes:

*   **`SadAttemptAtNetwork.m & EEGPreProcessing.m`**: Modular MATLAB scripts used for repeatable experiments and parameter tracking . Includes implementations for:
    *   Data preprocessing and filtering .
    *   EEG signal segmentation .
    *   CNN model architecture and training .
    *   Evaluation tooling (including confusion matrices and majority voting) .
*   **`DDF Poster Aaron Alymann Jeyaraj.pdf`**: The official poster presented at the University Research Symposium .
*   **`DDF Aaron Alymann Jeyaraj_EEG Biometric.pdf`**: The slide deck utilized to present the findings, methodology, and neural network architectures to an academic audience.

## 🛠 Methodology
1.  **Signal Processing:** Applied temporal filtering and artifact removal to raw 14-channel EEG streams to isolate relevant cortical activity .
2.  **Segmentation:** Windowed the continuous EEG streams into discrete epochs suitable for deep learning ingestion .
3.  **Classification:** Engineered a Convolutional Neural Network tailored for time-series biometric data to classify subject identities .

## 👨‍💻 About the Researcher
**Aaron Alymann Jeyaraj**
I am an incoming MS Computer Science student with a strong focus on applied machine learning and data infrastructure. Beyond conducting research, I have dedicated two years to serving as a Teaching Assistant and CS Tutor . This instructional background drives my commitment to writing clean, modular code and creating clear documentation, ensuring that complex technical systems remain accessible and reproducible for other researchers and students.
