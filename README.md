# Equipment Anomaly Detection

![Equipment Anomaly Detection](https://img.shields.io/badge/ML-Equipment%20Anomaly%20Detection-blue)
![Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

A machine learning application for predicting equipment failures before they occur, utilizing operational data from industrial equipment to identify potential anomalies and prevent downtime.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Information](#model-information)
- [Screenshots](#screenshots)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## Overview

This application analyzes equipment sensor data to identify patterns that indicate potential failures before they occur. By leveraging machine learning models trained on historical data, the system can provide early warnings about equipment that might require maintenance or inspection.

The project uses a Random Forest classification model that identifies anomalies based on several key parameters:
- Temperature
- Pressure
- Vibration
- Humidity
- Equipment type
- Location

## Features

- **Project Overview**: Summary of the project, key findings, and methodology
- **Interactive Data Exploration**: Dynamic visualizations of feature relationships and equipment fault patterns
- **Real-time Prediction**: Input equipment parameters to get instant fault predictions
- **Batch Prediction**: Upload CSV files with multiple equipment readings for bulk predictions
- **Feature Importance Analysis**: Understand which parameters have the most influence on fault prediction
- **Performance Metrics**: Clear visualization of model performance metrics

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository or download the source code:
   ```bash
   git clone <repository-url>
   cd archive
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have the following files in the directory:
   - `equipment_anomaly_data.csv`: Dataset with equipment readings
   - `equipment_fault_prediction_model.pkl`: Trained machine learning model
   - `equipment_fault_prediction_scaler.pkl`: Fitted scaler for data preprocessing

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. The application will open in your default web browser at `http://localhost:8501`

3. Navigate through the application using the sidebar:
   - **Project Overview**: Learn about the project and its findings
   - **Data Exploration**: Analyze equipment data through interactive visualizations
   - **Prediction Model**: Make predictions using the trained model

4. For single predictions:
   - Select equipment parameters using the sliders and dropdowns
   - Click "Predict Fault Probability" to get results

5. For batch predictions:
   - Prepare a CSV file with equipment parameters
   - Upload the file in the Batch Prediction tab
   - Download the results with fault predictions

## Project Structure

```
archive/
│
├── app.py                              # Streamlit application
├── requirements.txt                    # Project dependencies
├── equipment_anomaly_data.csv          # Dataset with equipment readings
├── equipment_fault_prediction_model.pkl # Trained machine learning model
├── equipment_fault_prediction_scaler.pkl # Fitted scaler for preprocessing
└── README.md                           # Project documentation
```

## Model Information

The fault prediction model is a Random Forest classifier trained on historical equipment data. The model achieves:

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.98  |
| Precision | 0.99  |
| Recall    | 0.95  |
| F1 Score  | 0.99  |

### Feature Importance
- Vibration: 32%
- Temperature: 24%
- Pressure: 18%
- Temperature-Pressure Ratio: 16%
- Humidity: 10%

## Screenshots

(Add screenshots of your application here)

## Dependencies

- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn
- joblib

## Contributing

Contributions to improve the application are welcome. Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request