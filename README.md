# question-duplicates-detection
A deep learning project that uses Siamese Networks with TensorFlow to detect duplicate or semantically similar questions using triplet loss and cosine similarity.
# ❓ Question Duplicate Detection using Siamese Networks

This project implements a **Siamese Neural Network** in **TensorFlow** to detect **duplicate questions** based on semantic similarity.  
Given two questions, the model determines whether they are asking the **same thing** or **different things**, even if they are phrased differently.

---

## 🚀 Project Overview

Duplicate question detection is a common problem in **Question-Answering Systems**, **search engines**, and **community forums** like **Quora** or **Stack Overflow**.

For example:

| Question 1                       | Question 2                           | Are They Duplicates? |
|--------------------------------|------------------------------------|----------------------|
| "How can I learn Python fast?" | "What is the fastest way to learn Python?" | ✅ Yes |
| "What is AI?"                 | "How to bake a cake?"             | ❌ No |

The model uses:
- **Siamese Networks** → Two identical neural networks sharing weights.
- **Triplet Loss** → Encourages similar questions to have closer embeddings.
- **Cosine Similarity** → Measures semantic closeness between embeddings.
- **TensorFlow & Keras** → To build, train, and evaluate the model.

---

## 🧠 Key Features

### **1. Data Preprocessing**
- Tokenizes and encodes questions into fixed-length sequences.
- Builds a vocabulary from the dataset.
- Handles out-of-vocabulary words with a special token.

### **2. Siamese Network Architecture**
- Uses **two identical subnetworks** to produce embeddings for each question.
- Computes similarity between embeddings using **cosine similarity**.
- Optimizes using **triplet loss** for better separation between duplicate and non-duplicate pairs.

### **3. Hard Negative Mining**
- During training, the model selects **challenging negative examples** (questions that look similar but are different).
- This improves the model’s ability to distinguish tricky duplicates.

### **4. Training & Evaluation**
- Trains on batches of paired questions.
- Evaluates performance using **accuracy, cosine similarity, and F1-score**.
- Supports testing on **custom user-defined questions**.

---

## 🏗️ Model Architecture


- **Embedding Layer:** Learns dense vector representations of words.
- **BiLSTM/GRU Layer:** Captures contextual relationships.
- **Dense Layer:** Transforms embeddings into fixed-size semantic vectors.
- **Cosine Similarity:** Measures similarity between the two question vectors.

---

## 📌 Implementation Details

### **1. Building the Siamese Model**
```python
import tensorflow as tf
from tensorflow.keras import layers

def create_siamese_network(vocab_size, embedding_dim=128, lstm_units=64):
    # Shared embedding & LSTM layers
    inputs = tf.keras.Input(shape=(None,))
    embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs)
    lstm = layers.Bidirectional(layers.LSTM(lstm_units))(embedding)
    dense = layers.Dense(128, activation="relu")(lstm)
    model = tf.keras.Model(inputs, dense)
    return model

# Create two identical subnetworks
siamese_model = create_siamese_network(vocab_size)
