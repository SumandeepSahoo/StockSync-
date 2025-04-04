# 📈 StockSync

**StockSync** is a full-stack web application that leverages machine learning to analyze stock market trends, visualize key indicators, and predict future stock movements using LSTM-based deep learning models.

---

## 🚀 Features

- 📊 Visualize stock trends (returns, volume, EMA, PSAR)
- 🔁 Analyze feature correlations and confusion matrix
- 🧠 Train LSTM model for directional prediction (Up/Down)
- 📉 Track model accuracy/loss across epochs
- 📈 Predict future trends and display R² Score

---

## 🧩 Tech Stack

### Frontend
- **React.js** – Interactive UI
- **Recharts** & **Heatmap-Grid** – Visualizations

### Backend
- **Flask** – API server
- **Pandas**, **NumPy**, **scikit-learn** – Data processing
- **TensorFlow / Keras** – LSTM model
- **yFinance** – Fetching historical stock data

---

## 📂 Project Structure

StockSync/
│
├── frontend/                # React App
│   ├── components/
│   ├── AnalysisCharts.jsx
│   └── ...
│
├── backend/                 # Flask Server
│   ├── app.py
│   ├── analysis_module.py
│   └── ...
│
└── README.md


---

## 📌 How It Works

1. User enters stock symbol and date range.
2. Data is fetched using **yFinance**.
3. Technical indicators like EMA and PSAR are calculated.
4. LSTM model is trained on derived features to predict stock direction.
5. Results and visualizations are rendered via React frontend.

---

## 🛠️ Setup Instructions

## Backend Setup (Flask + ML)
 
cd backend
pip install -r requirements.txt
python app.py

Make sure you’re using Python 3.8+ and have virtualenv activated.

## Frontend Setup (React)

cd frontend
npm install
npm start

📈 Future Improvements

📌 Add multi-stock comparison
🧾 Export results to PDF/CSV
🔑 User login with dashboard
📅 Custom intervals and notifications
📉 Live data and price alerts

📦 Requirements

See the full list in requirements.txt. Key packages:
Flask
pandas
numpy
tensorflow
scikit-learn
yfinance

🧑‍💻 Author

Made with ❤️ by Sumandeep.


