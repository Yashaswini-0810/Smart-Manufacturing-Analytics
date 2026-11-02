# Smart Manufacturing Production Analytics and Performance Dashboard

An interactive data analytics and machine learning dashboard designed to analyze manufacturing operations, monitor production performance, and predict manufacturing job status.

## Project Overview

Manufacturing industries generate large amounts of operational data related to machines, production activities, energy consumption, processing time, and job performance.

This project uses data analytics and machine learning techniques to transform manufacturing data into meaningful insights. It provides an interactive dashboard to monitor production performance, analyze machine health, understand job status distribution, explore relationships between operational parameters, and predict the status of prospective manufacturing jobs.

The dashboard is developed using Python and Streamlit, with interactive visualizations powered by Plotly.

## Objectives

- Analyze manufacturing production data.
- Monitor machine performance and availability.
- Evaluate production efficiency and processing time.
- Analyze completed, delayed, and failed manufacturing jobs.
- Explore relationships between manufacturing parameters.
- Develop a machine learning model to predict manufacturing job status.
- Present operational insights through an interactive dashboard.

## Project Workflow

```text
Manufacturing Dataset
        |
        v
    Data Loading
        |
        v
    Data Cleaning
        |
        v
  Data Preprocessing
        |
        v
Exploratory Data Analysis
        |
        v
Machine & Production Analysis
        |
        v
  Machine Learning
        |
        v
 Performance Dashboard
        |
        v
Insights & Decision Support
```

## Dataset Description

The project uses a manufacturing dataset containing 1,000 records and 13 columns.

The dataset includes information related to:

- Machine identification
- Operation type
- Material usage
- Processing time
- Energy consumption
- Machine availability
- Scheduled and actual production timestamps
- Optimization category
- Manufacturing job status

The target variable used for machine learning is **Job_Status**, which contains three classes:

- Completed
- Delayed
- Failed

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data visualization |
| Scikit-learn | Machine learning and preprocessing |
| Joblib | Saving and loading the trained model |
| Plotly | Interactive dashboard visualizations |
| Streamlit | Dashboard development and deployment |
| Google Colab | Data analysis and model development |
| GitHub | Source code and project management |
| Streamlit Community Cloud | Dashboard deployment |

## Dashboard Features

The dashboard contains six major sections:

### 1. Executive Dashboard
- Displays important manufacturing performance indicators.
- Provides an overall summary of production operations.
- Presents key metrics through visualizations.

### 2. Machine Health
- Analyzes machine availability.
- Provides insights into machine performance.
- Helps identify machine-level operational trends.

### 3. Production Analysis
- Analyzes production activities.
- Examines processing time and energy consumption.
- Provides machine-wise and operation-wise comparisons.

### 4. Quality & Job Status
- Analyzes completed, delayed, and failed jobs.
- Displays job status distribution.
- Helps understand manufacturing performance patterns.

### 5. Correlation Analysis
- Explores relationships between numerical manufacturing parameters.
- Uses correlation analysis to understand associations between variables.

### 6. Job Status Prediction
- Provides an interactive prediction interface.
- Accepts six manufacturing input features:
  - Machine ID
  - Operation Type
  - Material Used
  - Processing Time
  - Energy Consumption
  - Machine Availability
- Uses a trained Random Forest classification pipeline to predict the manufacturing job status.

Optimization Category is excluded from the prediction inputs because its relationship with the final job status raised a potential target-leakage concern.

## Machine Learning Approach

The project implements a supervised machine learning approach using **Random Forest Classification**.

The workflow includes:

1. Data preprocessing.
2. Feature selection.
3. Categorical feature encoding.
4. Train-test splitting using stratification.
5. Hyperparameter tuning using GridSearchCV.
6. Model evaluation using classification metrics.
7. Saving the trained model using Joblib.
8. Integrating the saved model into the Streamlit dashboard.

The trained model is used to demonstrate the prediction of manufacturing job status based on operational input parameters.

## Project Structure

```text
Smart-Manufacturing-Analytics/
│
├── app.py
├── requirements.txt
├── README.md
│
├── manufacturing_cleaned.csv
├── manufacturing_original.csv
│
└── best_model_final.joblib
```

### File Description

| File | Description |
|---|---|
| `app.py` | Main Streamlit dashboard application |
| `requirements.txt` | Required Python dependencies |
| `manufacturing_cleaned.csv` | Cleaned manufacturing dataset |
| `manufacturing_original.csv` | Original manufacturing dataset |
| `best_model_final.joblib` | Saved machine learning pipeline |
| `README.md` | Project documentation |

## Installation and Execution

Follow these steps to run the project locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Yashaswini-0810/Smart-Manufacturing-Analytics.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Smart-Manufacturing-Analytics
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```

The dashboard will open in your browser.

## Deployment

The dashboard is deployed using **Streamlit Community Cloud** and connected to the GitHub repository.

The application can be accessed through its deployed Streamlit URL.

## Future Enhancements

- Improve prediction performance through additional feature engineering and model experimentation.
- Integrate real-time manufacturing data.
- Develop advanced machine health monitoring capabilities.
- Explore predictive maintenance techniques.
- Add automated alerts for production delays and machine failures.
- Integrate the dashboard with industrial IoT data sources.

## Conclusion

The Smart Manufacturing Production Analytics and Performance Dashboard demonstrates how data analytics and machine learning can be applied to manufacturing operations.

By combining data preprocessing, exploratory analysis, machine learning, and interactive visualizations, the project provides a unified platform for understanding production performance and exploring manufacturing job status predictions.

The project aims to support data-driven analysis and better operational decision-making in smart manufacturing environments.

---

**Project Domain:** Smart Manufacturing | Data Analytics | Machine Learning

**Developed using:** Python, Machine Learning, and Streamlit
