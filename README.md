# 🚦 Few-Shot Traffic Sign Recognition Using Siamese Networks

## Project Overview

### What is This Project?
A machine learning system that recognizes traffic signs using **Few-Shot Learning** 
and **Siamese Neural Networks**. Unlike traditional approaches requiring thousands 
of images, our system learns from just **5-10 images per class**.

### Problem Statement
Traditional deep learning models require large datasets (1000s of images per class) 
to achieve good accuracy. This is:
- Time-consuming to collect
- Expensive to label
- Not practical for real-world scenarios

Our solution: Learn from minimal data using Siamese Networks that compare image similarity.

### Why Few-Shot Learning?
- **Real-world analogy**: Humans learn to recognize signs from just a few examples
- **Practical**: Limited data availability in many domains
- **Efficient**: Reduces data collection and labeling costs
- **Scalable**: Easy to add new sign types without retraining

### How Siamese Networks Work
A Siamese Network consists of two identical neural networks that:
1. Take two images as input
2. Extract features from each image independently
3. Compare the extracted features
4. Output a similarity score (0 = different, 1 = identical)

The network learns by training on pairs:
- **Similar pairs** (same traffic sign) → Pushes similarity score toward 1
- **Dissimilar pairs** (different signs) → Pushes similarity score toward 0

## Dataset

### GTSRB (German Traffic Sign Recognition Benchmark)
- **Total Classes**: 43 different traffic sign types
- **Our Selection**: 5 representative classes
  1. Speed limit 30 km/h (Class 1)
  2. Speed limit 50 km/h (Class 2)
  3. Speed limit 80 km/h (Class 5)
  4. Stop sign (Class 14)
  5. No entry (Class 17)

### Few-Shot Dataset Composition
- **Total Images**: 40 (only 8 per class!)
- **Training Split**: 6 images per class
- **Reference Split**: 2 images per class
- **Image Specifications**: 32×32 RGB pixels
- **Format**: Standardized and normalized

### Data Pairs Generated
- **Similar Pairs**: 75 pairs (same class images)
- **Dissimilar Pairs**: 75 pairs (different class images)
- **Total Training Pairs**: 150
- **Dataset Split**: 
  - Training: 70% (105 pairs)
  - Validation: 15% (22 pairs)
  - Testing: 15% (23 pairs)

## Project Phases

### Phase 1: Data Preparation
**Objective**: Clean and standardize the dataset


### Phase 2: Pair Generation
**Objective**: Create training data for Siamese Network


### Phase 3: Model Architecture
**Objective**: Design Siamese Network for similarity learning

### Phase 4: Model Training
**Objective**: Train Siamese Network on generated pairs

## Technical Stack

- **Deep Learning**: TensorFlow/Keras
- **Data Processing**: NumPy, OpenCV, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Version Control**: Git/GitHub
- **Language**: Python 3.8+