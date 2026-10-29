<img src="assets/icon.svg" width="64" height="64" alt="Brain Data Explorer Icon">

# Brain Data Explorer (braindb)

An interactive visual analytics dashboard built with **Streamlit**, **Pandas**, **Plotly**, and **Scikit-learn** to explore, analyze, and cluster brain-related clinical datasets.

> ℹ️ **Disclaimer**: Educational and research demonstration project only. This dashboard does not provide medical diagnoses.

---

## 🌟 Key Features

- **📊 Dataset Overview**: Instant metric cards tracking total records, columns, missing values, and duplicate rows.
- **🔍 Dynamic Filtering**: Multi-select categorical filters and numeric range sliders to narrow down cohorts and metrics, with instant CSV export.
- **📈 Interactive Visualizations**:
  - Histograms (metric distributions)
  - Scatter plots with categorical color coding
  - Bar charts (grouped cohort averages)
  - Correlation heatmaps across measurements
- **📉 Statistical Analysis**: Summary statistics (`describe()`), missing-value breakdowns by column, and duplicate detection.
- **🤖 Machine Learning (K-Means Clustering)**: Unsupervised clustering using standard-scaled features to identify data clusters and download labeled CSVs.

---

## 📂 Project Structure

```
├── app.py                           # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── synthetic_brain_study_report.csv # Sample 250-record dataset for testing & demo
├── start.command                    # macOS launcher script
├── assets/
│   └── icon.svg                     # App icon
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or 3.11 recommended.

### 2. Installation
Clone the repository:
```bash
git clone https://github.com/thisisvaishnav/baindb.git
cd baindb
```

Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Testing with Sample Data

You can upload your own brain dataset CSV, or upload the included [synthetic_brain_study_report.csv](synthetic_brain_study_report.csv) containing 250 clinical participant records (Healthy Controls, Mild Cognitive Impairment, Alzheimer's Disease) with biomarkers such as hippocampal volume, cortical thickness, functional connectivity, and cognitive scores.
