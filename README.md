# Comparative Study of Traditional Machine Learning, Deep Learning, and Transformer-Based Models for Text Classification
## Project Overview
This project focuses on a comparative analysis of different text representation techniques and machine learning models for a multi-class text classification task (Yes / No / Maybe).

We explore traditional methods such as TF-IDF and Word2Vec, as well as contextual embeddings generated using BERT. These representations are evaluated using multiple classifiers including Logistic Regression, Support Vector Machine (SVM), XGBoost, and an Artificial Neural Network (ANN). Additionally, a fine-tuned BERT model is used to assess the performance of transformer-based approaches.

The study aims to understand how different feature extraction methods and models influence classification performance on the same dataset.
## Objectives

- To compare TF-IDF, Word2Vec, and BERT-based text representations.
- To evaluate the performance of classical machine learning models such as Logistic Regression, SVM, and XGBoost.
- To implement and assess a deep learning-based Artificial Neural Network (ANN).
- To fine-tune a pretrained BERT model for text classification.
- To analyze the impact of contextual embeddings versus traditional embeddings.
- To compare all models using standard evaluation metrics such as accuracy, precision, recall, and F1-score.

## Dataset

This project uses the **PubMedQA dataset**, a biomedical question answering dataset designed for research in natural language processing within the medical domain. The dataset is available at: https://pubmedqa.github.io/

PubMedQA consists of questions derived from PubMed abstracts, along with corresponding answers and contextual evidence. Each sample typically includes a biomedical question and a label indicating the answer type, commonly structured as **Yes / No / Maybe** for classification tasks.

The dataset is widely used for evaluating models on scientific and biomedical reasoning tasks, making it suitable for testing both traditional machine learning methods and advanced transformer-based models like BERT.

In this project, the dataset is used for multi-class text classification after appropriate preprocessing, where textual inputs are converted into numerical representations using TF-IDF, Word2Vec, and BERT embeddings before being passed to different classification models.

## Methodology

The project follows a structured pipeline for text classification:

### 1. Exploratory Data Analysis (EDA)
- Analysis of class distribution (Yes / No / Maybe)
- Inspection of text length variations
- Basic statistical and visual analysis of the dataset

### 2. Feature Extraction
- TF-IDF representation for sparse feature extraction
- Word2Vec embeddings for semantic representation
- BERT-based embeddings for contextual representation

### 3. Model Training
- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost
- Artificial Neural Network (ANN)

### 4. Fine-Tuning
- Pretrained BERT model is fine-tuned on the labeled dataset for improved contextual understanding and classification performance

### 5. Evaluation
- Models are evaluated and compared using standard performance metrics such as accuracy, precision, recall, and F1-score

## Evaluation Metrics

The performance of all models is evaluated using the following metrics:

- Accuracy: Measures the proportion of correctly classified samples.
- Precision: Measures the correctness of positive predictions.
- Recall: Measures the ability of the model to identify all relevant instances.
- F1-Score: Harmonic mean of precision and recall, used for balanced evaluation.

These metrics are used to compare the effectiveness of different embeddings and models across the classification task.
