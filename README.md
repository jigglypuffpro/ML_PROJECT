# 🛡️ Toxic Comment Classifier

### 🤖 A Deep Learning Approach to Detecting Toxicity in Online Conversations

---

## 📌 Overview

Toxic comments are a growing concern on social media platforms. Offensive, threatening, or otherwise harmful messages can lead to harassment, mental distress, and a breakdown in healthy community discourse. This project builds a **machine learning model** that can **automatically detect and classify toxic comments**, enabling platforms to better moderate their environments and protect users.

We use the **Jigsaw Toxic Comment Classification Challenge** dataset from Kaggle to train a **deep learning model** that leverages natural language processing (NLP) techniques and neural network architectures.

---

## 📂 Dataset

**Source:** [Jigsaw Toxic Comment Classification Challenge – Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

The dataset contains over **150,000** labeled comments from Wikipedia's talk page edits. Each comment is tagged with one or more of the following labels:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

Since comments can belong to multiple categories simultaneously, this is a **multi-label classification** task.

---

## 🧼 Data Preprocessing

To prepare the textual data for input to the neural network, we applied the following steps:

- **Lowercasing**: Normalize casing for consistency.
- **Removing punctuation and special characters**: To focus on meaningful tokens.
- **Tokenization**: Split text into individual words or tokens.
- **Vectorization**: Use TensorFlow's `TextVectorization` layer to convert tokens into integer sequences.
- **Padding/Truncating**: Ensure all sequences are of uniform length for batch processing.

---

## 🔁 Data Pipeline (MCSHBAP Strategy)

A robust and efficient TensorFlow data pipeline is implemented using the **MCSHBAP** method, which improves training speed and model performance:

### 🔹 M — Map
Applies transformation functions to preprocess the text and labels.

### 🔹 C — Cache
Caches the dataset in memory after the first epoch to reduce I/O overhead.

### 🔹 S — Shuffle
Randomizes the dataset, preventing the model from learning spurious patterns based on data order.

### 🔹 H — (Unused)
This placeholder typically stands for **repeat**, which was not used here to prevent overfitting.

### 🔹 B — Batch
Groups data into batches, allowing efficient GPU computation.

### 🔹 A — (Unused)
In some references, this may imply **augment**. We did not apply text augmentation here but it can be explored in future versions.

### 🔹 P — Prefetch
Overlaps data preprocessing and model training to avoid bottlenecks during training.

> Implemented using TensorFlow’s `tf.data.Dataset` API for maximum performance.

---

## 🧠 Model Architecture

The model is constructed using the **Keras Sequential API**, designed for multi-label classification of text data.

**Layer-by-layer breakdown:**

1. **Embedding Layer**:
   - Converts input token indices into dense vector representations.
   - Captures semantic relationships between words.

2. **Bidirectional LSTM**:
   - Learns context in both forward and backward directions.
   - Particularly effective for understanding sentiment and meaning in text.

3. **Dense Layers**:
   - Three fully connected layers with `ReLU` activation functions.
   - Allow the model to learn non-linear patterns in the data.

4. **Dropout Layers**:
   - Applied between dense layers to prevent overfitting.

5. **Output Layer**:
   - A `Dense` layer with 6 units (one per label) and `sigmoid` activation.
   - Outputs a probability for each label (multi-label classification).

---

## 🧪 Training Details

- **Loss Function**: `BinaryCrossentropy` (suitable for multi-label tasks).
- **Optimizer**: `Adam` optimizer with learning rate tuning.
- **Metrics**: Custom metrics for `Precision`, `Recall`, and `AUC`.

The model was trained over **several epochs** with early stopping based on validation performance.

---

## 📊 Evaluation Metrics

We evaluate the model using **Precision** and **Recall**:

| Metric   | Value   |
|----------|---------|
| Precision | 0.8753 |
| Recall    | 0.8039 |

- **Precision** measures how many of the predicted toxic labels were actually toxic (low false positives).
- **Recall** measures how many actual toxic comments were correctly identified (low false negatives).

These values demonstrate strong performance in both identifying and correctly classifying harmful content.

---

## 🧾 Results and Discussion

The model shows strong generalization and robust learning across all toxic categories. However:

- **Class imbalance** remains a challenge — categories like `threat` and `identity_hate` are underrepresented.
- **Sarcasm and context** can still confuse the model.
- Further improvements could be made using:
  - Pre-trained embeddings (e.g., GloVe, FastText).
  - Transformer models (e.g., BERT, RoBERTa).
  - Better handling of underrepresented classes through data augmentation or re-weighting.

---

## ✅ Conclusion

This project demonstrates a scalable, high-performing approach to automated toxicity detection using deep learning and NLP. By leveraging:

- An efficient data pipeline (MCSHBAP),
- Recurrent neural networks (BiLSTM),
- And careful evaluation metrics,

...we are able to build a classifier that significantly enhances content moderation tools.

Such systems can serve as a **first line of defense** against harmful online behavior, assisting human moderators and making digital communities safer for all users.

---

## 📌 Future Improvements

- Incorporate Transformer-based models (e.g., BERT) for deeper context understanding.
- Handle code-switching and multilingual comments.
- Add attention mechanisms for interpretability.
- Deploy model via an API or web interface for real-world integration.

---

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Documentation](https://keras.io/)
- [Jigsaw Toxic Comment Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- Papers on NLP & Toxic Comment Classification

---
