# Bert-Hate-Speech-Identification

## 📌 Project Overview

This project detects hate speech in social media text using Machine Learning and Deep Learning techniques. It includes text preprocessing, feature extraction using TF-IDF, a Logistic Regression model, and a BERT model for accurate text classification.

## 🚀 Features

- Hate Speech Detection
- Text Preprocessing
- TF-IDF Vectorization
- Logistic Regression Model
- BERT Model
- Confusion Matrix Visualization
- Custom Text Prediction

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Transformers (BERT)
- PyTorch
- Flask
- HTML
- Git
- GitHub
  
## 📂 Dataset

This project uses the **Hate Speech and Offensive Language Dataset** from Kaggle.

**Dataset Source:**
https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

The dataset contains labeled Twitter posts classified into hate speech, offensive language, and neither. It is used to train and evaluate machine learning and deep learning models for hate speech detection.

## 📁 Project Structure

```text
Bert-Hate-Speech-Identification/
│
├── dataset/
├── model/
├── results/
│   └── confusion_matrix.png
├── templates/
├── app.py
├── train_model.py
├── train_bertmodel.py
├── README.md
├── LICENSE
└── .gitignore
```

## 📊 Model Performance

- Logistic Regression Accuracy: **90%+**
- BERT Model: Fine-tuned for hate speech classification
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Confusion Matrix generated and stored in `results/confusion_matrix.png`

## ▶️ How to Run the Project

1. Clone the repository:
```bash
git clone https://github.com/saipavithira/Bert--Hate-Speech-Identification.git
```

2. Navigate to the project folder:
```bash
cd Bert--Hate-Speech-Identification
```

3. Install the required libraries:
```bash
pip install -r requirements.txt
```

4. Run the Flask application:
```bash
python app.py
```

## 🚀 Future Improvements

- Improve prediction accuracy using larger datasets.
- Deploy the project on the cloud.
- Add multilingual hate speech detection.
- Build a modern web interface.
- Optimize the BERT model for faster inference.

## 👩‍💻 Author

Saipavithira M

GitHub: https://github.com/saipavithira

LinkedIn: https://www.linkedin.com/in/saipavithira-manimaran-0597702b2

---

⭐ If you found this project useful, consider giving it a star.
