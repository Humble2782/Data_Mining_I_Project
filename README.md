Predicting Injury Severity in Road Accidents

A Real-Time Classification Approach (Data Mining I Project)

📖 Overview

This project was developed by students at the University of Mannheim as part of the Data Mining I curriculum. The goal is to enhance modern vehicle telematics (such as the European eCall system) by integrating a machine learning classifier capable of predicting injury severity immediately after an accident.

Current eCall systems transmit location and passenger count but lack injury severity data: a critical gap for emergency triage. Using historical data from the French National Interministerial Observatory for Road Safety (ONISR), we developed a pipeline to classify accidents into three severity levels: Uninjured, Lightly Injured, and Severe (Hospitalized/Dead) based solely on real-time variables.

Read the full detailed analysis: Project Report, Project Presentation

🏗️ Repository Structure

├── ETL/               # Modularized pipeline merging yearly tables (characteristics, locations, vehicles, users)
├── dashboard/         # Real-time Streamlit dashboard
├── data/              # Dataset in different preprocessing stages and final training/testing sets
├── documents/         # Project report and presentation
├── exploration/       # Notebooks for EDA and Clustering (K-Prototypes)
├── models/            # Training scripts for CatBoost, Balanced Random Forest, and Ridge
└── .gitignore         # Git ignore configuration

📊 Dataset & Feature Engineering

We utilized the ONISR "Bulletins d’Analyse des Accidents Corporels" (BAAC) dataset (2019-2023), processing over 600,000 records. The final unit of analysis is the individual user.

Key Engineering Challenges

To convert raw database dumps into a model-ready format, we implemented complex preprocessing logic:

Vehicle Antagonist Resolution: In multi-vehicle accidents, we developed an algorithm to identify the specific "opposing" vehicle (Antagonist) that caused the injury, calculating an impact_delta based on the mass difference between vehicles (e.g., Bicycle vs. HGV).

Location Deduplication: Implemented a completeness_score to resolve duplicate location entries, prioritizing records with rich metadata (Road Category, Speed Limit).

Road Complexity Index: A composite score (0-10) aggregating intersection type, lane count, and traffic regime to quantify environmental risk.

Cyclical Time Features: Sine/Cosine transformations for hours and months to capture temporal patterns.

🚀 Methodology

The project follows the CRISP-DM lifecycle, focusing on handling the significant class imbalance (only 16% severe injuries).

1. Clustering (Accident Personas)

We used K-Prototypes (handling mixed categorical/numerical data) to identify 5 distinct accident personas, such as:

Cluster 1: Night-time accidents involving young adults (18-30) in low visibility.

Cluster 3: High-complexity urban intersection accidents.

2. Classification Models

We evaluated three distinct architectures against a speed-limit baseline:

Ridge Classifier (RC): A linear baseline with L2 regularization and random undersampling.

Balanced Random Forest (BRF): An ensemble method that undersamples the majority class during bootstrapping.

CatBoost (CB): A gradient boosting algorithm chosen for its native handling of high-cardinality categorical features.

🏆 Results

CatBoost was selected as the optimal model, achieving the highest F1-Macro score and Cohen's Kappa.

Model

Precision (Severe)

Recall (Severe)

F1-Macro

CatBoost

0.46

0.73

0.66

Balanced RF

0.47

0.70

0.66

Ridge Classifier

0.40

0.77

0.61

Baseline

0.16

0.08

0.33

Critical Success: The system successfully identifies ~96% of severe cases as at least "injured," meeting the primary safety objective of not missing critical cases.

Key Predictors: The most important features identified were mobile_obstacle_struck, impact_delta, and type_of_collision.

🛠️ Getting Started

Prerequisites

Python 3.8+

pip

Installation & Usage

This project is built to run as a Dashboard Application.

Clone the repository:

git clone [https://github.com/Humble2782/Data_Mining_I_Project](https://github.com/Humble2782/Data_Mining_I_Project)
cd Data_Mining_I_Project

Run the Dashboard:
Navigate to the dashboard folder and run the application. Dependencies should be installed from this directory.

cd dashboard
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py

👥 Authors

David Cebulla (dcebulla@mail.uni-mannheim.de)

Gabriel Himmelein (ghimmele@mail.uni-mannheim.de): Website

Lukas Ott (lott@mail.uni-mannheim.de)

Artur Loreit (arloreit@mail.uni-mannheim.de)

Aaron Niemesch (aniemesc@mail.uni-mannheim.de)

Submitted to the Data and Web Science Group at the University of Mannheim.

📄 License

This project is licensed under the MIT License.
