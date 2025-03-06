# Deep Learning Model Classifies Human-written Poetry and AI-generated Poetry

This repository contains a deep learning model that classifies poetry as either human-written or AI-generated. The model is built on the PhoBERT architecture and fine-tuned on a custom dataset. It provides scripts for training, testing, and running the model to recognize and differentiate between the two types of poetry.

## Repository Contents

- **README.md**: This file.
- **data_test.rar**: Compressed archive containing the test dataset.
- **data_train_val.rar**: Compressed archive containing the training and validation dataset.
- **datacl.py**: Python module for collection data.
- **run recognize potry.py**: Script to run the poetry recognition model on new input.
- **test_phobert.ipynb**: Jupyter Notebook for testing and evaluating the model.
- **train_phobert.py**: Script for fine-tuning the PhoBERT model on the training dataset.

## Overview

The model is based on the PhoBERT-base model (~110 million parameters) and is fine-tuned to distinguish between:
- **Human-written poetry**
- **AI-generated poetry**

The goal is to provide insights into the subtle differences between human and AI literary styles, as well as to evaluate the model's performance on this challenging classification task.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/DuyhocAI/Deep-Learning-Model-Classifies-human-written-poetry-and-AI-generated-poetry.git
   cd Deep-Learning-Model-Classifies-human-written-poetry-and-AI-generated-poetry
   
Install Dependencies:

Ensure you have Python 3.7+ installed. Then install the required packages. If a requirements.txt is not provided, you can install the following packages manually:

bash
Sao chép
pip install torch transformers tqdm matplotlib scikit-learn
Data Preparation
data_train_val.rar: Contains the training and validation datasets.
data_test.rar: Contains the test dataset.
Extract these archives into appropriate folders before running the training or testing scripts. The scripts expect the data to be organized into subfolders (e.g., human and AI).

Usage
Training the Model
To fine-tune the PhoBERT model on your training data, run:


python train_phobert.py

This script loads the training and validation data from the extracted data_train_val folder, fine-tunes the model, and saves the best model and tokenizer to the specified output directory.

Testing / Running Recognition
To classify new poetry samples or evaluate the model on the test set, run:

python "run recognize potry.py"

Alternatively, you can open and run the Jupyter Notebook test_phobert.ipynb for an interactive evaluation.

Acknowledgements
This project leverages the PhoBERT model from vinai/phobert-base and builds upon recent advances in deep learning for natural language processing and computational creativity.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For any questions or suggestions, please contact [zestdapoet@gmail.com].
