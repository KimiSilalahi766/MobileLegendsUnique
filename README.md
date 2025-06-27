# Mobile Legends Player Clustering Analysis

## Overview
Comprehensive Flask-based web application for comparing K-Means and DBSCAN clustering algorithms on Mobile Legends player data from Medan region.

# MobileLegendsUnique

🎮 A web app to analyze and predict Mobile Legends heroes using machine learning (Flask + scikit-learn).

🚀 **Live App:**  
👉 [Click here to open the website](https://mobilelegends-analyzer.up.railway.app)

## Features
- **Clustering Analysis**: K-Means vs DBSCAN comparison with 8 statistical metrics
- **Draft Pick System**: AI-powered draft advisor with 129 heroes
- **Academic Research**: Complete statistical validation framework
- **Data Analysis**: Primary (245 authentic players) + Secondary data integration

## Installation

1. Install Python 3.11+
2. Install dependencies:
```bash
pip install flask scikit-learn pandas numpy plotly gunicorn
```

3. Run the application:
```bash
python main.py
```

## Project Structure
```
├── app.py                      # Main Flask application
├── main.py                     # Entry point
├── clustering_engine.py        # K-Means & DBSCAN implementation
├── comparative_analysis.py     # Algorithm comparison module
├── statistical_validator.py    # Statistical validation framework
├── data_processor.py          # Data preprocessing
├── visualization.py           # Plotly visualization engine
├── draft_pick_system.py       # Mobile Legends draft system
├── research_framework.py      # Academic research tools
├── secondary_data_processor.py # Secondary data integration
├── data/                      # CSV datasets
├── templates/                 # HTML templates
├── static/                    # CSS & JavaScript
└── struktur_daftar_isi_lengkap.md # Thesis structure guide
```

## Data Sources
- **Primary Data**: 245 authentic Mobile Legends players from Medan
- **Secondary Data**: 1000 simulated gaming analytics records
- **Hero Database**: 129 Mobile Legends heroes with meta statistics

## Academic Metrics Implemented
1. Silhouette Score
2. Calinski-Harabasz Index
3. Davies-Bouldin Index
4. ANOVA F-Test
5. Kruskal-Wallis Test
6. Cross-validation Stability
7. Feature Importance F-ratio
8. Inter-cluster Distance Analysis

## Research Validity: 92% - Ready for Thesis Defense

## Usage
1. Access main dashboard at `http://localhost:5000`
2. Compare clustering algorithms in real-time
3. Use Draft Pick System for Mobile Legends strategy
4. Access Academic Research tools for statistical validation

Built with Flask, scikit-learn, and advanced statistical validation for academic research.

## ⚠️ License & Usage

This project is under a custom license.  
You are **not allowed** to reuse, reupload, or distribute this code for your own use **without explicit permission** from the author.

Please contact me for collaboration or permission.
