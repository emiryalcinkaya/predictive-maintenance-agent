# Predictive Maintenance Agent

A predictive maintenance system that predicts machine failures from industrial sensor data using a Decision Tree classifier and generates maintenance recommendations with Llama 3.

**Developed by Emir Yalçınkaya**

---

## Overview

This project demonstrates how a machine learning model can be integrated into an AI agent for predictive maintenance.

The agent:

- Predicts machine failures from sensor data
- Estimates prediction confidence
- Generates maintenance recommendations using Llama 3

---

## Features

- Decision Tree classifier
- Random Forest model comparison
- Feature importance analysis
- Model evaluation
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Maintenance recommendations using Llama 3

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**

Source:
- UCI Machine Learning Repository

Input Features

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

Target

- Machine Failure
  - 0 = No Failure
  - 1 = Failure

---

## Machine Learning Pipeline

1. Load dataset
2. Analyze the data
3. Preprocess the data
4. Apply One-Hot Encoding
5. Train the Decision Tree model
6. Compare with Random Forest
7. Evaluate model performance
8. Use the selected model in the AI agent

---

## Project Structure

```text
predictive-maintenance-agent/
│
├── data/
├── images/
├── models/
├── src/
│   ├── agent.py
│   ├── analysis.py
│   ├── data_loader.py
│   ├── evaluate_model.py
│   ├── feature_importance.py
│   ├── llm_advisor.py
│   ├── model_utils.py
│   ├── preprocessing.py
│   └── train_model.py
│
├── requirements.txt
└── README.md
```

---

## Technologies

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- Ollama
- Llama 3

---

## Model Performance

### Decision Tree (Selected Model)

| Metric | Score |
|---------|------:|
| Accuracy | 97.85% |
| Precision | 68.66% |
| Recall | 67.65% |
| F1-score | 68.15% |

### Random Forest

| Metric | Score |
|---------|------:|
| Accuracy | 96.85% |
| Precision | 52.58% |
| Recall | 75.00% |
| F1-score | 61.82% |

The Decision Tree model was selected because it achieved the best overall performance.

---

## Run

Train the models

```bash
python src/train_model.py
```

Run the AI agent

```bash
python src/agent.py
```

---

## Workflow

```text
Sensor Data
      │
      ▼
Decision Tree
      │
      ▼
Failure Prediction
      │
      ▼
Llama 3
      │
      ▼
Maintenance Recommendation
```