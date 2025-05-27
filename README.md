# MBTI_Personality_prediction
Predict Myers-Briggs (MBTI) personality types based on user-written text using Natural Language Processing (NLP) and Machine Learning techniques.
#  MBTI Personality Prediction using NLP

Predict Myers-Briggs (MBTI) personality types from users' written posts using Natural Language Processing and Machine Learning techniques.

---

##  Overview

This project focuses on classifying users into one of the 16 MBTI personality types based on their textual data. By leveraging NLP techniques and machine learning models, we attempt to identify patterns in user posts that correspond to specific personality traits.

---

## 🗂️ Dataset

- **Source**: [Kaggle - MBTI Personality Type Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type)
- **Format**: CSV
- **Columns**:
  - `type`: One of 16 MBTI personality types (e.g., INFP, ESTJ, etc.)
  - `posts`: A string of 50 user posts concatenated with "|||"

---

## ⚙️ Technologies Used

| Library/Tool             | Description |
|--------------------------|-------------|
| `pandas`, `numpy`        | Data handling and manipulation |
| `matplotlib`, `seaborn`  | Data visualization |
| `nltk`, `re`             | Text preprocessing and tokenization |
| `scikit-learn`           | Feature extraction (TF-IDF), model training, evaluation |
| `WordCloud`              | Visualization of key terms |

---

##  Preprocessing Steps

- Combined 50 posts into a single string per user
- Removed URLs, punctuation, and special characters
- Converted text to lowercase
- Removed stopwords using NLTK
- Generated cleaned text for training

---

##  Model Pipeline

1. **TF-IDF Vectorization**: Convert cleaned text into numerical vectors.
2. **Train/Test Split**: Divide the dataset into training and testing sets.
3. **Model Training**: Use `Logistic Regression` for classification.
4. **Evaluation**:
   - Classification report (Precision, Recall, F1-Score)
   - Confusion matrix for analysis

---

##  Visualizations

- Distribution of MBTI personality types
- WordClouds for common terms used by each type
- Confusion Matrix

---

## 📈 Results

Initial results using Logistic Regression show reasonable performance, particularly on more frequent MBTI types. Further model optimization or use of transformers like BERT can significantly improve results.

---

##  How to Run

1. Clone the repository
2. Install dependencies from `requirements.txt`
3. Open and run the Jupyter notebook
4. Optionally download the dataset from Kaggle and place it in the project directory

---

##  Future Enhancements

- BERT-based modeling for better semantic understanding
- Use of ensemble models for improved accuracy
- Multilabel classification (separating the 4 MBTI dichotomies: I/E, N/S, F/T, J/P)

---

##  Author

Anubhav Singh 
singhanubhav9415@gmail.com


---



