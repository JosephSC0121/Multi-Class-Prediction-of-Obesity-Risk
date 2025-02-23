# Multiclass Obesity Risk Prediction

A **Streamlit** application utilizing **XGBoost** to predict obesity risk based on multiple individual characteristics. The project includes data preprocessing, feature distribution visualization, model training, and performance evaluation.

## ✨ Features
- **Exploratory data analysis** with key visualizations.
- **Automated preprocessing** for categorical and numerical data.
- **Optimized XGBoost model** for multiclass classification.
- **Interactive Streamlit interface** for real-time predictions.
- **Results export** to CSV files.

## 📂 Project Structure

```plaintext
.  
├── data/          # Training and test datasets  
│   ├── train.csv  
│   └── test.csv  
├── models/        # Trained model storage  
├── scripts/       # Modularized code  
│   ├── preprocess.py  # Data preprocessing  
│   ├── train.py       # Model training  
│   ├── evaluate.py    # Model evaluation  
│   └── predict.py     # Prediction on new data  
├── main.py        # Streamlit interface  
├── requirements.txt  # Project dependencies  
└── README.md      # Documentation  
```

## 🚀 Installation & Execution

### 1️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 2️⃣ Run the Application
```sh
streamlit run main.py
```

## 📊 Visualization

The application enables visualization of feature distributions, real-time predictions, and model accuracy analysis.


## 🎥 Video Demonstration
[Watch the Video](https://www.youtube.com/watch?v=VJgwfG208Vk)



