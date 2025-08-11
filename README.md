## Plant Disease Classifier using SVM
## Overview
This project is a Machine Learning Flask web application that classifies plant diseases based on given input features using the Support Vector Machine (SVM) algorithm.
It helps farmers and agricultural researchers identify plant diseases quickly, enabling timely treatment.

## Features
Trained on 50 plant disease dataset samples

Uses Support Vector Machine (SVM) for classification

Lightweight Flask web application for easy deployment

User-friendly HTML interface for predictions

Clean, modern light theme UI

## Project Structure
```
plant_disease_classifier/
│
├── dataset.csv              # Plant disease dataset (50 samples)
├── train.py                 # ML model training script
├── app.py                   # Flask application script
├── templates/
│   └── index.html           # Frontend HTML page
├
│           
├── svm_model.pkl            # Saved trained model
├── requirements.txt         # Required Python dependencies
└── README.md                # Project documentation
```
## Installation
## 1️ Clone the Repository
```
git clone https://github.com/yourusername/plant-disease-classifier-svm.git
cd plant-disease-classifier-svm
```
## 2️ Install Dependencies
```
pip install -r requirements.txt
```
## 3️ Train the Model
```
python train.py
```
## 4️ Run the Flask App
```
python app.py
```
## Usage
```
Open your browser and go to http://127.0.0.1:5000
```

Enter plant details (features like leaf color, size, spots, texture, etc.)

Click Predict to see if the plant is healthy or diseased.

## Machine Learning Model
Algorithm: Support Vector Machine (SVM)

Kernel: RBF (Radial Basis Function)

Purpose: Classify plant condition into Healthy or Diseased

Input: Numerical features related to plant health

Output: Binary classification (0 = Healthy, 1 = Diseased)

## Dataset
The dataset (dataset.csv) contains 50 entries with plant feature values and corresponding labels.
Example:

Leaf_Color	Leaf_Size	Spots	Moisture_Level	Disease_Label
4.5	6.2	2	7.8	1
5.1	7.0	0	8.0	0

## Requirements

scikit-learn
pandas
numpy
## Screenshot
![alt text](<Screenshot 2025-08-11 115457.png>)
![alt text](<Screenshot 2025-08-11 113714.png>)