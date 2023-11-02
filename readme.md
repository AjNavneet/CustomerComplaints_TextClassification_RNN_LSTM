# Text Classification with RNN and LSTM Models on Customer Complaints Data

### Business Overview

Text classification is a fundamental application of Natural Language Processing (NLP). Neural network-based methods, especially RNNs and LSTMs, have made significant strides in various NLP tasks. In this project, we explore the application of Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models for text classification. These models are well-suited for handling sequential data and have demonstrated their effectiveness in various NLP tasks.

---

### Aim

The primary goal of this project is to perform text classification using RNN and LSTM models on a dataset of customer complaints about consumer financial products.

---

### Data Description

The dataset comprises more than two million customer complaints about consumer financial products. It includes columns for the actual text of the complaint and the product category associated with each complaint. Pre-trained word vectors from the GloVe dataset (glove.6B) are used to enhance text representation.

---

### Tech Stack

- Language: `Python`
- Libraries: `pandas`, `torch`, `nltk`, `numpy`, `pickle`, `re`, `tqdm`, `sklearn`

---

## Approach

1. Install the necessary packages using the `pip` command.
2. Import the required libraries.
3. Define configuration file paths.
4. Process GloVe embeddings:
   - Read the text file.
   - Convert embeddings to a float array.
   - Add embeddings for padding and unknown items.
   - Save embeddings and vocabulary.
5. Process Text data:
   - Read the CSV file and remove null values.
   - Handle duplicate labels.
   - Encode the label column and save the encoder and encoded labels.
6. Data Preprocessing:
   - Convert text to lowercase.
   - Remove punctuation.
   - Eliminate digits.
   - Remove additional spaces.
   - Tokenize the text.
7. Build Data Loader.
8. Model Building:
   - RNN architecture.
   - LSTM architecture.
   - Train function.
   - Test function.
9. Model Training:
   - RNN model training.
   - LSTM model training.
10. Prediction on Test Data.

---

## Modular Code Overview

1. **Input**: Contains data required for analysis, including:
   - `complaints.csv`
   - `glove.6B.50d.txt` (download it from [here](https://nlp.stanford.edu/projects/glove/))

2. **Source**: Contains modularized code for various project steps, including:
   - `model.py`
   - `data.py`
   - `utils.py`

   These Python files contain helpful functions used in the `Engine.py` file.

3. **Output**: Contains files required for model training, including:
   - `embeddings.pkl`
   - `label_encoder.pkl`
   - `labels.pkl`
   - `model_lstm.pkl`
   - `model_rnn.pkl`
   - `vocabulary.pkl`
   - `tokens.pkl`
   
   (The `model_lstm.pkl` and `model_rnn.pkl` files are our saved models after training)

4. **config.py**: Contains project configurations.

5. **Engine.py**: The main file to run the entire project, which trains the model and saves it in the output folder.

---

## Key Concepts Explored

1. Understanding the concept of pre-trained word vectors.
2. Introduction to Recurrent Neural Networks (RNN).
3. Understanding how RNN networks operate.
4. Exploring the vanishing gradient problem.
5. Introduction to Long Short-Term Memory (LSTM) networks.
6. Understanding LSTM network functionality.
7. Steps to process GloVe embeddings.
8. Preparing data for the models.
9. Handling spaces and digits.
10. Punctuation removal.
11. Creating data loaders for RNN and LSTM models.
12. Building RNN models.
13. Building LSTM models.
14. Training RNN and LSTM models using GPU or CPU.
15. Making predictions on new text data.

---