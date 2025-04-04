import React, { useState } from "react";
import AnalysisCharts from "./AnalysisCharts";

function App() {
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [symbol, setSymbol] = useState("");
  const [loading, setLoading] = useState(false);
  const [responseData, setResponseData] = useState(null);
  const [error, setError] = useState(null);

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResponseData(null);
    const requestData = { startDate, endDate, symbol };

    try {
      const res = await fetch("http://localhost:5000/api/run_analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestData),
      });
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || "Failed to process request");
      }
      const data = await res.json();
      setResponseData(data);
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center bg-gray-50 py-10 px-6">
      <div className="max-w-6xl w-full grid grid-cols-1 md:grid-cols-2 gap-10">
        {/* Left Side: Form */}
        <div className="chart-container">
        <div className="chart-card">
        <h2 className="chart-title">Stock Market Prediction</h2>
          <form onSubmit={handleFormSubmit} className="space-y-5">
            <div>
              <label className="block text-gray-700 font-medium mb-2">
                Start Date
              </label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="input-field"
                required
              />
            </div>
            <div>
              <label className="block text-gray-700 font-medium mb-2">
                End Date
              </label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="input-field"
                required
              />
            </div>
            <div>
              <label className="block text-gray-700 font-medium mb-2">
                Stock Symbol (e.g. AAPL, TSLA)
              </label>
              <input
                type="text"
                placeholder="Enter stock symbol"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                className="input-field"
                required
              />
            </div>
            <button
              type="submit"
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg transition-transform transform hover:scale-105"
              disabled={loading}
            >
              {loading ? "Processing..." : "Submit & Analyze"}
            </button>
          </form>
          {error && (
            <div className="mt-4 bg-red-100 text-red-700 p-3 rounded-md text-center">
              {error}
            </div>
          )}
        </div>
        </div>

        {/* Right Side: About Project */}
        <div className="chart-card">
        <div className="bg-blue-600 text-white shadow-lg rounded-lg p-8 flex flex-col justify-center">
          <h2 className="text-3xl font-semibold mb-4 ">About This Project</h2>
          <p className="text-lg opacity-90">
            This Stock Market Analysis tool helps users analyze stock trends using machine learning.
            It fetches real-time stock data from Yahoo Finance and applies LSTM-based models to predict
            future trends.
          </p>
          <p className="mt-4 opacity-90">
            Built with **React.js**, **Flask**, and **LSTM-based deep learning model**.
          </p>
        </div>
        </div>
      </div>

      {/* Charts/Graphs Below */}
      <div className="max-w-6xl w-full mt-10">
        {responseData && (
          <AnalysisCharts
            chartData={responseData.data_head}
            trainAccuracy={responseData.trainAccuracy}
            valAccuracy={responseData.valAccuracy}
            trainLoss={responseData.trainLoss}
            valLoss={responseData.valLoss}
            confusionMatrix={responseData.confusion_matrix}
            correlationMatrix={responseData.correlation_matrix}
            correlationLabels={responseData.correlation_labels}
          />
        )}
      </div>
    </div>
  );
}

export default App;
