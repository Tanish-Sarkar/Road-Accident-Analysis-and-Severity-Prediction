# 🚗 Road Accident Analysis and Severity Prediction

> A Data Science Capstone Project that analyzes Indian road accident data to uncover patterns, identify risk factors, and predict accident severity using machine learning models (Decision Tree & SVM).

---

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![ML Models](https://img.shields.io/badge/Models-DecisionTree%20%7C%20SVM-orange)
![Made with ❤️](https://img.shields.io/badge/Made%20with-%F0%9F%92%9C-red)

---

## 📘 Project Overview

This project explores **road accident severity prediction** using real-world datasets from the **Ministry of Road Transport and Highways (MoRTH)** and **Kaggle**.  
It applies **Exploratory Data Analysis (EDA)** and **Machine Learning** to understand how environmental, temporal, and infrastructural factors influence accident outcomes.

---

## 🧭 Table of Contents

1. [Abstract](#1-abstract)  
2. [Problem Statement](#2-problem-statement)  
3. [Objectives](#3-objectives)  
4. [Dataset Description](#4-dataset-description)  
5. [Methodology](#5-methodology)  
6. [Exploratory Data Analysis (EDA)](#6-exploratory-data-analysis-eda)  
7. [Machine Learning Models](#7-machine-learning-models)  
8. [Results and Discussion](#8-results-and-discussion)  
9. [Ethical Considerations](#9-ethical-considerations)  
10. [Conclusion and Recommendations](#10-conclusion-and-recommendations)  
11. [References](#11-references)

---

## 1. Abstract

This project analyzes Indian road accident data to uncover patterns, identify high-risk conditions, and predict the severity of accidents. It uses government-sourced datasets to perform **exploratory data analysis (EDA)** and **machine learning classification** using **Decision Tree** and **SVM** models.

By understanding factors such as weather, lighting, road type, and time of occurrence, this project aims to provide actionable insights for improving road safety and policymaking.

---

## 2. Problem Statement

Despite government initiatives, India records one of the world’s highest numbers of road fatalities annually.  
This project seeks to:

- Identify the factors influencing accident severity.  
- Analyze and visualize historical data.  
- Build predictive models to estimate severity based on environmental and infrastructural features.

---

## 3. Objectives

- Analyze large-scale Indian road accident data (MoRTH/Kaggle).  
- Identify the impact of road, weather, and environmental conditions.  
- Visualize spatial and temporal patterns.  
- Build classification models (Decision Tree, SVM) for severity prediction.  
- Provide actionable insights to improve road safety.

---

## 4. Dataset Description

**Source:** [Kaggle – India Road Accident Dataset](https://www.kaggle.com/datasets/data125661/india-road-accident-dataset)

The dataset contains three CSV files:

- `AccidentsBig.csv`: Accident-level details (time, date, location, severity, weather, etc.)  
- `VehiclesBig.csv`: Vehicle-level data (type, speed, maneuver, etc.)  
- `CasualtiesBig.csv`: Victim-level data (age, gender, severity)  

**Total Records:** ~60,000  

**Key Columns:**  
`accident_severity`, `road_type`, `light_conditions`, `weather_conditions`, `speed_limit`, `urban_or_rural_area`, `date`, `latitude`, `longitude`

---

## 5. Methodology

1. **Data Acquisition:** Dataset downloaded from Kaggle.  
2. **Data Cleaning:** Removed duplicates, handled missing values, standardized column names.  
3. **Feature Engineering:** Extracted time-based features (year, month, day).  
4. **Data Merging:** Combined the three CSVs on `Accident_Index`.  
5. **EDA:** Visualized severity, environmental, and spatial patterns.  
6. **Modeling:** Trained Decision Tree and SVM classifiers for severity prediction.  
7. **Evaluation:** Compared models using Accuracy, F1-score, and Confusion Matrix.

---

## 6. Exploratory Data Analysis (EDA)

### 6.1 Accident Severity Distribution
- Most accidents are **slight**, while **fatal** accidents are least frequent.  
- Class imbalance observed — important consideration for modeling.
 <img src="https://github.com/Tanish-Sarkar/Road-Accident-Analysis-and-Severity-Prediction/blob/main/outputs/1.png" alt="" width="500" height="400">
 <img src="https://github.com/Tanish-Sarkar/Road-Accident-Analysis-and-Severity-Prediction/blob/main/outputs/2.png" alt="" width="500" height="400">

### 6.2 Temporal Analysis
- Accidents peak on **weekends** and **daytime hours**.  
- Seasonal spikes during **monsoon** and **festival months**.
 <img src="https://github.com/Tanish-Sarkar/Road-Accident-Analysis-and-Severity-Prediction/blob/main/outputs/3.png" alt="" width="500" height="400">
 <img src="https://github.com/Tanish-Sarkar/Road-Accident-Analysis-and-Severity-Prediction/blob/main/outputs/4.png" alt="" width="500" height="400">
 <img src="https://github.com/Tanish-Sarkar/Road-Accident-Analysis-and-Severity-Prediction/blob/main/outputs/5.png" alt="" width="500" height="400">

### 6.3 Environmental Factors
- Clear weather dominates in frequency but not necessarily safety.  
- **Night/no-light conditions** increase fatal severity.  
- **Wet surfaces** lead to fewer but more severe accidents.
 <img src="https://github.com/Tanish-Sarkar/Road-Accident-Analysis-and-Severity-Prediction/blob/main/outputs/6.png" alt="" width="500" height="400">
 <img src="https://github.com/Tanish-Sarkar/Road-Accident-Analysis-and-Severity-Prediction/blob/main/outputs/7.png" alt="" width="500" height="400">
 <img src="https://github.com/Tanish-Sarkar/Road-Accident-Analysis-and-Severity-Prediction/blob/main/outputs/8.png" alt="" width="500" height="400">

### 6.4 Road Type & Area Analysis
- **Single carriageways** and **urban areas** have higher accident frequency.  
- **Rural roads** have fewer but more fatal outcomes.  
- **High speed limits** correlate with increased severity.
 <img src="https://github.com/Tanish-Sarkar/Road-Accident-Analysis-and-Severity-Prediction/blob/main/outputs/9.png" alt="" width="500" height="400">
 <img src="https://github.com/Tanish-Sarkar/Road-Accident-Analysis-and-Severity-Prediction/blob/main/outputs/10.png" alt="" width="500" height="400">
 <img src="https://github.com/Tanish-Sarkar/Road-Accident-Analysis-and-Severity-Prediction/blob/main/outputs/11.png" alt="" width="500" height="400">
 <img src="https://github.com/Tanish-Sarkar/Road-Accident-Analysis-and-Severity-Prediction/blob/main/outputs/12.png" alt="" width="500" height="400">

### 6.5 Geospatial Hotspot Analysis
- Heatmaps show dense clusters near **metro cities**.  
- **National highways** exhibit continuous high accident density.
 <img src="https://github.com/Tanish-Sarkar/Road-Accident-Analysis-and-Severity-Prediction/blob/main/outputs/13.png" alt="" width="500" height="400">

---

## 7. Machine Learning Models

### 7.1 Feature Selection
**Features:**  
`day_of_week`, `road_type`, `speed_limit`, `light_conditions`,  
`weather_conditions`, `road_surface_conditions`, `urban_or_rural_area`  

**Target:** `accident_severity`

---

### 7.2 Algorithms Used

#### 🟩 Decision Tree Classifier
- Recursive partitioning using Gini Index/Entropy.
- Max depth: **8** (to prevent overfitting).  
- Key influential features: `speed_limit`, `light_conditions`, `road_type`.

**Pros:**
- Easy to interpret and visualize.  
- Reveals causal feature relationships.  
- Useful for policy-level insights.  

**Cons:**
- Prone to overfitting on small variations.

---

#### 🟦 Support Vector Machine (SVM)
- Implemented with **RBF kernel** to handle non-linear decision boundaries.  
- **Parameters:**  
  - Kernel = RBF  
  - C = 1  
  - Gamma = scale  

**Pros:**
- High accuracy and generalization.  
- Handles non-linear and high-dimensional data.  
- Less prone to overfitting.

**Cons:**
- Computationally expensive for large datasets.  
- Difficult to interpret (black-box).

---

### 7.3 Model Comparison

| Criteria | Decision Tree | SVM (RBF) |
|-----------|----------------|------------|
| Interpretability | ✅ Very High | ⚪ Low |
| Performance | ⚪ Moderate–High | ✅ High |
| Overfitting Risk | ⚠️ High (if unpruned) | ✅ Low |
| Training Speed | ✅ Fast | ⚪ Slower |
| Feature Scaling | Not Needed | Required |
| Use Case | Model explanation & policy | High-accuracy prediction |

---

### 7.4 Model Evaluation

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|---------|-----------|
| Decision Tree | ~85% | 0.84 | 0.83 | 0.83 |
| SVM (RBF) | ~88% | 0.87 | 0.86 | 0.86 |
  <img src="https://github.com/Tanish-Sarkar/Road-Accident-Analysis-and-Severity-Prediction/blob/main/outputs/14.png" alt="" width="500" height="400">
  <img src="https://github.com/Tanish-Sarkar/Road-Accident-Analysis-and-Severity-Prediction/blob/main/outputs/15.png" alt="" width="500" height="400">


---

### 7.5 Observations

- **SVM** outperformed Decision Tree with better F1-score (~0.86).  
- **Class imbalance** led to confusion between *Serious* and *Slight* cases.  
- **Top features:** Speed limit, light conditions, road type, and junction detail.  
- Decision Tree: Highly interpretable for policymakers.  
- SVM: Strong predictive accuracy for production use.

---

## 8. Results and Discussion

- Accidents show strong **temporal** and **spatial** patterns.  
- **Speed**, **lighting**, and **junction complexity** are key severity factors.  
- Predictive models show potential for **real-time risk classification** and **proactive safety planning**.

---

## 9. Ethical Considerations

- Maintain **data anonymity**; avoid personal identification.  
- Ensure **responsible use** of data visualizations.  
- Use models for **policy support**, not liability assessment.  
- Acknowledge **regional biases** or underreporting in datasets.

---

## 10. Conclusion and Recommendations

### 🧩 Conclusion
Accident severity is influenced primarily by **road type**, **lighting**, and **speed**.  
SVM achieved the **best performance** (~88% accuracy).

### 💡 Recommendations
- Improve **road lighting** and **signage**.  
- Enforce **speed control** on high-risk areas.  
- Redesign **junction layouts** for safety.  
- Increase **public awareness** during peak accident hours.

---

## 11. References

- Ministry of Road Transport and Highways (MoRTH) — *Road Accident Reports*  
- [Kaggle: India Road Accident Dataset](https://www.kaggle.com/datasets/data125661/india-road-accident-dataset)  
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)  
- [Seaborn & Matplotlib Documentation](https://matplotlib.org/)

---

## 🌐 Connect & Acknowledge

**Author:** [Tanish Sarkar](https://github.com/)  
**Professor:** Dr. Deepak Kumar Verma  
**Department:** Computer Engineering  
**Institution:** Faculty of Engineering and Technology

---

> ✨ *“Data doesn’t just tell stories — it saves lives when analyzed with purpose.”*
