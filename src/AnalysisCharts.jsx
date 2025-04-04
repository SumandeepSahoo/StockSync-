import React from "react";
import './app.css';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, Legend, BarChart, Bar
} from "recharts";
import HeatMapGrid from "react-heatmap-grid"; // for heatmap visualizations

const AnalysisCharts = ({
  chartData = [],
  trainAccuracy = [],
  valAccuracy = [],
  trainLoss = [],
  valLoss = [],
  confusionMatrix = [],
  correlationMatrix = [],
  correlationLabels = []
}) => {
  const dailyReturnData = chartData.map(item => ({
    Date: item.Date,
    Return: item.Return,
  }));

  const trendData = chartData.map(item => ({
    Date: item.Date,
    Trend: item.Trend
  }));

  const volumeData = chartData.map(item => ({
    Date: item.Date,
    Volume: item.Volume
  }));

  const epochData = trainAccuracy.map((val, idx) => ({
    epoch: idx + 1,
    trainAcc: trainAccuracy[idx],
    valAcc: valAccuracy[idx],
    trainLoss: trainLoss[idx],
    valLoss: valLoss[idx]
  }));

  const renderCorrelationHeatmap = () => {
    if (!correlationMatrix.length || !correlationLabels.length) {
      return <p>No correlation data available.</p>;
    }
    return (
      <div className="chart-card">
        <h2 className="chart-title">Feature Correlations</h2>
        <div style={{ width: "500px" }}>
          <HeatMapGrid
            xLabels={correlationLabels}
            yLabels={correlationLabels}
            data={correlationMatrix}
            cellStyle={(background, value) => ({
              background: `rgba(0, 132, 255, ${Math.abs(value)})`,
              fontSize: "14px",
              color: "#000"
            })}
            cellRender={(value) => (value != null ? value.toFixed(2) : "")}
          />
        </div>
      </div>
    );
  };

  

  return (
    <div className="chart-container">
      <div className="chart-card">
        <h2 className="chart-title">Daily Return Over Time</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={dailyReturnData}>
            <XAxis dataKey="Date" />
            <YAxis />
            <Tooltip />
            <CartesianGrid stroke="#ccc" />
            <Line type="monotone" dataKey="Return" stroke="#00cc00" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-card">
        <h2 className="chart-title">Trend Over Time</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trendData}>
            <XAxis dataKey="Date" />
            <YAxis />
            <Tooltip />
            <CartesianGrid stroke="#ccc" />
            <Line type="monotone" dataKey="Trend" stroke="#800080" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-card">
        <h2 className="chart-title">Volume Over Time</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={volumeData}>
            <XAxis dataKey="Date" />
            <YAxis />
            <Tooltip />
            <CartesianGrid stroke="#ccc" />
            <Bar dataKey="Volume" fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      </div>

     


    </div>
  );
};

export default AnalysisCharts;
