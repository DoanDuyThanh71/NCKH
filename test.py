import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from pyvi import ViTokenizer
import re

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\r', ' ', text)
    text = text.lower()
    text = " ".join(text.split())  # Tách từ bằng khoảng trắng
    return text

# Load mô hình đã được huấn luyện
svm_model = joblib.load('svm_model.pkl')

# Load vectorizer đã được huấn luyện
vectorizer = joblib.load('vectorizer.pkl')

# Hàm dự đoán nhãn của câu
def predict_label(sentence):
    # Tiền xử lý câu
    preprocessed_sentence = preprocess_text(sentence)
    # Chuyển đổi câu thành vector
    sentence_vector = vectorizer.transform([preprocessed_sentence])
    # Dự đoán nhãn của câu
    predicted_label = svm_model.predict(sentence_vector)
    return predicted_label[0]

# Nhập và dự đoán liên tục cho đến khi gặp từ 'exit'
while True:
    # Nhập vào một câu từ người dùng
    input_sentence = input("Nhập vào một câu (hoặc 'exit' để kết thúc): ")
    
    # Kiểm tra điều kiện để thoát vòng lặp
    if input_sentence.lower() == 'exit':
        break
    
    # Dự đoán và in ra nhãn của câu
    predicted_label = predict_label(input_sentence)
    print("Nhãn của câu:", predicted_label)
