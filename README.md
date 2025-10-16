# NLP Sentiment Analysis of Product Reviews

This project analyzes product reviews to classify them as either positive or negative sentiment. It explores various Natural Language Processing (NLP) techniques, from classic machine learning models to a Recurrent Neural Network (RNN).

---

## 🚀 Project Overview

The goal of this project is to build and evaluate different models for sentiment classification. The process includes:
1.  **Text Pre-processing**: Cleaning and preparing the raw review text for analysis.
2.  **Vectorization**: Converting text into numerical features using Bag-of-Words, TF-IDF, and Word2Vec.
3.  **Model Training**: Implementing and comparing multiple classification models.
4.  **Evaluation**: Measuring the accuracy and performance of each model.

---

## 🛠️ Techniques and Libraries Used
* **Data Manipulation**: Pandas
* **Text Pre-processing**: NLTK
* **Vectorization**: Scikit-learn (CountVectorizer, TfidfVectorizer), Gensim (Word2Vec)
* **Machine Learning Models**:
    * Logistic Regression
    * K-Nearest Neighbors (KNN)
    * Naive Bayes
    * Recurrent Neural Network (RNN) with TensorFlow/Keras
* **Visualization**: Matplotlib, WordCloud

---

## ✨ Key Results

The performance of the models varied significantly based on the vectorization technique.

| Model                  | Vectorization | Test Accuracy |
| ---------------------- | ------------- | :-----------: |
| K-Nearest Neighbors    | Word2Vec      |     94.0%     |
| Naive Bayes            | Word2Vec      |     88.0%     |
| **Recurrent Neural Network (RNN)** | **Embedding Layer** | **96.0%** |

The **Recurrent Neural Network (RNN) achieved the highest accuracy of 96.0%**, demonstrating its strength in understanding sequential text data for sentiment classification.

### Negative Sentiment Word Cloud
![Negative Word Cloud](link_to_your_image_if_you_upload_it)

### Positive Sentiment Word Cloud (TF-IDF)
![Positive Word Cloud](link_to_your_image_if_you_upload_it)

*(To add images, you can take screenshots of your word clouds, upload them to the repository, and then reference them here.)*

---

## ⚙️ How to Run This Project

1.  Clone the repository:
    ```sh
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    ```
2.  Navigate to the project directory:
    ```sh
    cd YOUR_REPOSITORY_NAME
    ```
3.  Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```
4.  Open and run the `sentiment_analysis_nlp.ipynb` notebook in a Jupyter environment.
