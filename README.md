# BERT-Based AI Content Classification

This project leverages a BERT-based deep learning model to classify text articles as either **AI-generated** or **human-written**. Using PyTorch and the Hugging Face `transformers` library, the project implements fine-tuning of a pre-trained BERT model for binary classification.

---

## Overview

The primary goal of this project is to classify text data into two categories:
- **AI-generated**
- **Human-written**

The workflow includes:
1. Tokenizing text data using a BERT tokenizer.
2. Defining a PyTorch dataset and data loader for text and labels.
3. Building and training a custom BERT-based classifier.
4. Evaluating the model using stratified cross-validation.
5. Saving the trained model for deployment or further analysis.

---

## Features

- **Pre-trained BERT Model:** Fine-tunes `bert-base-uncased` for text classification.
- **Custom Dataset Class:** Implements a PyTorch-compatible dataset class for efficient data handling.
- **Cross-Validation:** Uses Stratified K-Fold cross-validation to ensure robust evaluation.
- **Evaluation Metrics:** Calculates accuracy, F1 score, precision, and recall.

---

## Requirements

- Python 3.7+
- Libraries:
  - `torch`
  - `transformers`
  - `pandas`
  - `numpy`
  - `sklearn`
## Future Improvements
- Experiment with advanced pre-trained models like RoBERTa or DeBERTa.
- Handle class imbalance with techniques like oversampling or weighted loss functions.
- Extend the model to support multiclass classification for other types of text.
