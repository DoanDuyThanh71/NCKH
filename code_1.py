
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from pyvi import ViTokenizer
import re

# Đọc dữ liệu
df = pd.read_excel('data_final.xlsx')

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\r', ' ', text)
    text = text.lower()
    text = ViTokenizer.tokenize(text)
    return text

# Tiền xử lý dữ liệu
df['patterns'] = df['patterns'].apply(preprocess_text)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(df['patterns'], df['tag'], test_size=0.2, random_state=42)

# Sử dụng TfidfVectorizer để chuyển đổi văn bản thành vector
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Định nghĩa các giá trị của các siêu tham số cần tinh chỉnh
param_grid = {
    'C': [0.01, 0.1,0.2,0.5, 0.8, 1, 1.2],      # Tham số penalty
    'kernel': ['linear', 'poly' ,'rbf','sigmoid'],  # Kernel function
}

# Khởi tạo một GridSearchCV object với mô hình SVM và các siêu tham số
grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5)

# Huấn luyện GridSearchCV trên tập huấn luyện
grid_search.fit(X_train, y_train)

# Lưu model đã huấn luyện


# In ra siêu tham số tốt nhất được tìm thấy
print("Best parameters:", grid_search.best_params_)

# In ra độ chính xác tốt nhất trên tập kiểm tra
print("Best accuracy:", grid_search.best_score_)

# Huấn luyện mô hình SVM với siêu tham số tốt nhất
best_clf = svm.SVC(**grid_search.best_params_)
best_clf.fit(X_train, y_train)
joblib.dump(grid_search, 'svm_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# In độ chính xác trên tập huấn luyện và tập kiểm tra của mô hình tốt nhất
print("Độ chính xác trên tập huấn luyện của mô hình tốt nhất: ", best_clf.score(X_train, y_train))
print("Độ chính xác trên tập kiểm tra của mô hình tốt nhất: ", best_clf.score(X_test, y_test))