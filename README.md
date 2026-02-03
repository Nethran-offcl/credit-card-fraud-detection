# 💳 Credit Card Fraud Detection System

An end-to-end machine learning web application that detects fraudulent credit card transactions using supervised learning techniques, with interactive visualizations and automated data preprocessing.

---

## 🚀 Features

- ✅ Detects fraudulent vs legitimate transactions
- ✅ Handles highly imbalanced datasets
- ✅ Automated data preprocessing & feature scaling
- ✅ Interactive web interface built with Streamlit
- ✅ Real-time prediction and visualization
- ✅ Clean and modular ML pipeline

---

## 🧠 Machine Learning Approach

### Algorithm
**Random Forest Classifier**

### Why Random Forest?
- Handles class imbalance well
- Robust to noise and outliers
- Strong performance on tabular financial data
- Provides feature importance insights

### Techniques Used
- Data normalization & preprocessing
- Train–test split
- Model evaluation using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python |
| **ML Libraries** | scikit-learn, NumPy, Pandas |
| **Web Framework** | Streamlit |
| **Visualization** | Matplotlib, Seaborn |

---

## 📊 Dataset

The project uses a credit card transaction dataset containing:
- Transaction features (anonymized for privacy)
- Class label (**Fraud** / **Not Fraud**)
- Highly imbalanced data (fraud cases are rare)

> ⚠️ **Note:** Dataset used only for educational and research purposes.

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Nethran-offcl/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 📈 Output

The application provides:
- 📊 Transaction classification results (Fraud/Legitimate)
- 📉 Visual insights into fraud vs non-fraud predictions
- 📋 Model performance metrics in real-time
- 🎯 Confusion matrix and classification reports

---

## 🎯 Use Cases

- 🏦 Financial fraud analysis
- 💰 Banking & payment systems
- 📚 Learning real-world ML deployment
- 🎓 Academic and portfolio projects

---

## 🔮 Future Improvements

- [ ] Add deep learning models (ANN / LSTM)
- [ ] Improve imbalance handling using SMOTE or ADASYN
- [ ] Deploy on cloud platforms (AWS / Render / Streamlit Cloud)
- [ ] Add transaction history tracking
- [ ] Implement ensemble methods (XGBoost, LightGBM)
- [ ] Add API endpoints for integration

---

## 👨‍💻 Author

**Nethran**  
GitHub: [@Nethran-offcl](https://github.com/Nethran-offcl)

---

## 📝 License

This project is open source and available for educational purposes.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Nethran-offcl/credit-card-fraud-detection/issues).

---
