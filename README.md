# Fake Job Posting Detection Using Machine Learning

> **Fake Job Posting Detection Using Machine Learning** is a project focused on using machine learning to identify whether a job posting is **genuine or fraudulent**.

## About the Project

Finding a genuine job online is not always easy. Some job postings can contain misleading information or may be created with the intention of tricking job seekers. This project explores how **machine learning can be used to analyze job postings and identify patterns that may be associated with fraudulent listings**. The basic idea is to take existing job-posting data, process the information, and train a machine learning model to learn the differences between genuine and fraudulent job postings.

## Dataset

For this project, I used the **Real / Fake Job Posting Prediction** dataset from Kaggle.

**Dataset:** [Real / Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

The dataset contains around **18,000 job descriptions**, including both real and fake job postings. It contains textual information as well as other information related to the job postings. The `fraudulent` column is used to identify whether a job posting is fraudulent.

Some of the important fields used in this project include:

* `title`
* `location`
* `department`
* `company_profile`
* `description`
* `requirements`
* `benefits`
* `fraudulent`

## How the Project Works

The project starts by loading the job-posting dataset and preparing the data for the machine learning model. Different text fields from each job posting are combined together, including the **job title, location, department, company profile, description, requirements, and benefits**. This gives the model more information about each job posting and allows it to learn from the overall text rather than relying on only one field.

## Text Processing

Since a machine learning model cannot directly understand normal text, the project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert the job-posting text into numerical features. TF-IDF helps represent the importance of words within the job-posting dataset. The resulting numerical features are then passed to the machine learning model.

## Machine Learning Model

For classification, I used **Logistic Regression**. The dataset is divided into training and testing data. The training data is used to teach the model, while the testing data is kept separate so that the model can be tested on job postings it has not seen during training. The model then predicts whether the job postings in the test data are fraudulent or genuine.

## Model Evaluation

After training, the model is evaluated using:

* **Accuracy**
* **Classification Report**
* **Confusion Matrix**

The classification report provides information such as precision, recall, and F1-score, while the confusion matrix gives a visual representation of the model's predictions.

## Saving the Model

After training, I also save the trained model and the TF-IDF vectorizer using `joblib`.

```text
fake_job_model.pkl
tfidf_vectorizer.pkl
```

The model file contains the trained Logistic Regression model, while the vectorizer file contains the TF-IDF setup used to convert text into numerical features.

This makes it possible to reuse them later without having to train the model again.

## Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Joblib**
* **TF-IDF**
* **Logistic Regression**

### Dataset Source

**Kaggle — Real / Fake Job Posting Prediction**

[Kaggle Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction?utm_source=chatgpt.com)
