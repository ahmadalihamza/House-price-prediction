# 🏠 California House Price Prediction App

A machine learning web application that predicts California median house prices using **Linear Regression**, **Ridge Regression**, and **Lasso Regression** models — built with **Streamlit**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📌 Project Overview

This project uses the **California Housing Dataset** to train and evaluate three regression models. Users can select a model, input house features, and get a predicted median house value in real-time via an interactive web UI.

---

## 🚀 Live Demo

> Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) (free hosting).

---

## 📂 Project Structure

```
├── app.py                       # Streamlit web application
├── linear_regression_model.py   # Model training script (Colab-exported)
├── linear_model.pkl             # Trained Linear Regression model
├── ridge_model.pkl              # Trained Ridge Regression model
├── lasso_model.pkl              # Trained Lasso Regression model
├── scaler.pkl                   # Fitted StandardScaler
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🧠 Models Used

| Model              | Description                                      |
|--------------------|--------------------------------------------------|
| Linear Regression  | Baseline OLS regression model                   |
| Ridge Regression   | L2 regularized regression (GridSearchCV tuned)  |
| Lasso Regression   | L1 regularized regression (GridSearchCV tuned)  |

---

## 🔧 Features Used

- `longitude`, `latitude`
- `housing_median_age`
- `total_rooms`, `total_bedrooms`
- `population`, `households`
- `median_income`
- `ocean_proximity` (one-hot encoded)
- **Engineered features:**
  - `rooms_per_household`
  - `bedrooms_per_room`
  - `population_per_household`

---

## 🖥️ Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/house-price-prediction-app.git
cd house-price-prediction-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your GitHub repo, branch (`main`), and set **Main file path** to `app.py`
5. Click **Deploy** — your app will be live in minutes!

---

## 📊 Model Performance (on test set)

| Model              | RMSE        | MAE         | R²    |
|--------------------|-------------|-------------|-------|
| Linear Regression  | ~68,000     | ~49,000     | ~0.63 |
| Ridge Regression   | ~67,500     | ~48,500     | ~0.64 |
| Lasso Regression   | ~68,000     | ~49,000     | ~0.63 |

> *Approximate metrics — retrain on your data for exact values.*

---

## 📷 App Screenshot

![App Screenshot](https://via.placeholder.com/800x400?text=California+House+Price+Predictor)

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — web UI
- **scikit-learn** — ML models
- **pandas / numpy** — data processing
- **matplotlib / seaborn** — visualizations (training notebook)

---

## 📃 License

This project is open source under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

Built with ❤️ using Python & Streamlit.
