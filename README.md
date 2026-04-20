# AI & Data Science Salary Analyzer

## 📌 Project Overview

This project is an **interactive salary analysis and prediction tool** for AI and Data Science professionals. It helps users understand salary trends across different industries, locations, experience levels, and education backgrounds.

**Target Users:** Job seekers, HR professionals, career advisors, and business students who want data-driven insights for salary negotiations and career planning.

## 🎯 Problem Statement

Job seekers often struggle to answer: *"What salary should I expect?"* This tool provides:
- Market analysis by industry, location, and experience
- Personalized salary predictions based on user profile
- Side-by-side comparison of different career options

## 📊 Dataset

- **Source:** Job Salary Prediction Dataset (Kaggle)
- **Size:** 250,000 records
- **Key Features:** job_title, experience_years, education_level, skills_count, industry, location, remote_work, salary

*Data accessed: April 2026*

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core analysis |
| Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| Scikit-learn | ML prediction model |
| Streamlit | Interactive web app |

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/salary-predictor.git
cd salary-predictor
```
### 2. Install dependencies
bash
pip install -r requirements.txt
### 3. Run the app
The app will open automatically at http://localhost:8501

``
### 📱 Features
Feature	Description
🔍 Data Filtering	Filter by industry, location, remote work
📊 Market Analysis	Salary by industry, experience, education
📈 Salary Distribution	Histogram with KDE curve
🎯 Salary Predictor	ML-based personalized predictions
⚖️ Compare Options	Compare two career paths
📥 Export Data	Download filtered data as CSV
🌐 Multi-language	English / Chinese support

### 📁 Project Structure
salary-predictor/
├── app.py                 # Streamlit application
├── analysis.ipynb         # Jupyter Notebook analysis
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── data/
    └── job_salary_prediction_dataset.csv



### 🔍 Key Findings
Experience matters most: 10+ years experience earns 80% more than entry-level

Education premium: PhD holders earn 25% more than Bachelor's degree holders

Remote work impact: Fully remote roles pay 15% more than on-site roles

Industry variance: Finance and Tech are the highest paying industries



### ⚠️ Limitations
Dataset primarily covers specific job markets (no China data)

Senior-level data (10+ years) is limited

Predictions are estimates, not guarantees



### 🤖 AI Disclosure
Tool	Version	Access Date	Usage
ChatGPT	GPT-4	April 2026	Code structure, debugging, documentation
DeepSeek	Latest	April 2026	Initial code generation, optimization
### 📹 Demo Video
[Link to demo video - to be added]

### 👤 Author
Student ID: [Your Student ID]

Course: ACC102 Mini Assignment

Track: Track 4 - Interactive Data Analysis Tool

### 📅 Submission Date
April 27, 2026
