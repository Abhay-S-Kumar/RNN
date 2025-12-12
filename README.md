# RNN


# 🌦️ Monthly Temperature Forecasting with RNN

This project implements a Recurrent Neural Network (SimpleRNN) using TensorFlow/Keras to predict monthly average temperatures based on historical weather data. The model is trained on a 10-year hourly weather dataset, resampled to monthly averages, and achieves high accuracy in forecasting future temperatures.

## 📊 Project Overview

  * **Goal:** Predict the average temperature of a specific month based on the weather patterns of the previous 12 months.
  * **Model:** SimpleRNN with Dropout regularization.
  * **Data Processing:** Resampling hourly data to monthly means, Feature Scaling (MinMax).
  * **Performance:** The model achieved an $R^2$ score of **\~0.92** on the test set.
  * **Forecasting:** Includes a module to generate recursive forecasts for future unseen months (7-step ahead prediction).

## 📂 Dataset

The dataset used is the **Weather Dataset** provided by `muthuj7` on Kaggle.

  * **Source:** [Kaggle - Weather Dataset](https://www.kaggle.com/datasets/muthuj7/weather-dataset)
  * **Content:** Hourly weather data from 2006 to 2016 including Temperature, Humidity, Wind Speed, Pressure, etc.
  * **Input Features Used:** `Temperature (C)`, `Apparent Temperature (C)`.

## 🛠️ Technologies Used

  * **Python 3.x**
  * **TensorFlow / Keras** (Deep Learning)
  * **Pandas & NumPy** (Data Manipulation)
  * **Scikit-Learn** (Preprocessing & Metrics)
  * **Matplotlib** (Visualization)
  * **KaggleHub** (Dataset downloading)

## ⚙️ Installation & Requirements

To run this notebook, ensure you have the following dependencies installed:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow kagglehub
```

## 🚀 Methodology

1.  **Data Loading:** The dataset is downloaded directly via `kagglehub`.
2.  **Preprocessing:**
      * The `Formatted Date` column is converted to datetime objects and set as the index.
      * Data is resampled from **Hourly** to **Monthly Averages**.
      * Data is scaled between 0 and 1 using `MinMaxScaler`.
3.  **Sequence Creation:**
      * The model uses a "look-back" window (`n_steps`) of **12 months** to predict the temperature of the 13th month.
4.  **Model Architecture:**
      * Input Layer (Sequence length: 12)
      * `SimpleRNN` Layer (64 units, ReLU activation)
      * `Dropout` Layer (0.2) to prevent overfitting
      * `Dense` Output Layer (1 unit)
5.  **Training:**
      * Optimizer: Adam
      * Loss: Mean Squared Error (MSE)
      * Epochs: 70
6.  **Evaluation:** Verified using MAE, RMSE, and R-Squared metrics.
7.  **Forecasting:** A recursive loop generates predictions for the next 7 months into the future.

## 📈 Results

The model demonstrates strong predictive capability on the test set:

| Metric | Value (Approx) |
| :--- | :--- |
| **Test Loss (MSE)** | 0.0054 |
| **Test MAE** | 0.0577 |
| **Test RMSE** | 0.0733 |
| **R-Squared** | **0.9248** |

### Visualizations Included

1.  **Training vs Validation Loss:** To check for overfitting.
2.  **Actual vs Predicted:** Scatter plot showing correlation.
3.  **Forecast Graph:** Visualizing the historical temperature curve connected to the future predicted steps.

## 🔮 Future Improvements

  * Implement LSTM or GRU layers to compare performance against SimpleRNN.
  * Include additional features like Humidity or Pressure in the input sequences.
  * Hyperparameter tuning (adjusting look-back period or hidden units).

## 🤝 Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License

This project is open-source and available under the MIT License.
