# L’Oréal Skincare Optimizer: Sustainable Multi-Label Classification

This repository contains a machine learning solution developed for a **L’Oréal Business Case**. The objective was to build a sustainable and efficient system to classify skin conditions based on beauty product descriptions.

## 📋 Project Overview

In a production environment, deploying massive Large Language Models (LLMs) for every request is computationally expensive and environmentally taxing. This project demonstrates a **Knowledge Distillation** approach: using an LLM-labeled dataset to train a compact, high-performance **XGBoost Classifier Chain** that is ready for edge deployment.

### Key Objectives:

* **Multi-Label Classification:** Predicting 11 distinct skin conditions (e.g., Acne, Fine Wrinkles, Sensitivity).
* **Sustainability:** Minimizing the carbon footprint () and model size.
* **Efficiency:** Maintaining a high F1-Micro score while reducing inference latency.

---

## 🛠️ Technical Stack & Architecture

### 1. The Algorithm: Classifier Chain + XGBoost

We implemented a `ClassifierChain` using `XGBClassifier` as the base estimator.

* **Why?** Skin conditions are often correlated (e.g., "Oily" and "Pores"). A Classifier Chain accounts for these dependencies by using the predictions of previous models in the chain as features for subsequent ones.
* **Base Model:** XGBoost with `hist` tree method and `cuda` acceleration for optimized training speed.

### 2. Feature Engineering

* **NLP Pipeline:** Text preprocessing (Lemmatization) followed by a **TF-IDF Vectorizer** (1,800 features).
* **Metadata Integration:** 33 additional engineered features scaled via `StandardScaler`.
* **Total Features:** 1,833.

### 3. Threshold Optimization

To maximize the **F1 Micro** score, we implemented a two-step optimization:

* **Per-label optimization:** Finding the best probability threshold for each condition.
* **Global fine-tuning:** A coordinate descent-style loop to adjust thresholds globally to account for label interactions.

---

## 📊 Performance & Sustainability Metrics

| Metric | Result |
| --- | --- |
| **F1 Micro** | **0.7249** |
| **F1 Macro** | 0.6865 |
| **Model Size** | **15.82 MB** |
| **Carbon Emissions** | **0.0009 g CO2** |
| **Hamming Loss** | 0.1036 |

> **Note:** Carbon emissions were tracked using the `CodeCarbon` library during the training and inference cycles to ensure environmental transparency.
> Note: In accordance with our agreement, the dataset used for this project is private property and remains confidential.

---

## 📂 Repository Structure

```text
├── LOREAL_FINAL_VERSION.ipynb  # Main development notebook
├── deployment_pack.pkl         # Compressed joblib file (Model, Scaler, Vectorizer)
├── reports/
│   ├── report.txt              # Final performance summary
│   ├── f1_scores.png           # Visual breakdown of per-label performance
│   └── emissions.csv           # Detailed carbon tracking data
└── requirements.txt            # Environment dependencies

```

---

## 🚀 Deployment

The model is saved as a `deployment_pack.pkl` (approx. 15.8MB), making it ideal for:

* **Serverless Functions:** Fast cold starts due to small size.
* **Mobile/Edge Devices:** Low memory footprint for on-device inference.

---

## 👥 The Team

* **Obinna**
* **Aymen**
* **Ahmad**
* **Elizaveta**

