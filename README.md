Oral Microbiome Profiling and Machine Learning for OSCC Detection
This repository contains all the code, data outputs, visualizations, and results associated with a microbiome-based machine learning project aimed at identifying microbial biomarkers for early detection of oral squamous cell carcinoma (OSCC).

🧪 Project Overview
Oral squamous cell carcinoma (OSCC) is a prevalent malignancy with increasing incidence globally. Recent studies suggest that dysbiosis of the oral microbiome may play a significant role in OSCC onset and progression. This project analyzes 16S rRNA sequencing data from four publicly available studies in the NCBI SRA to identify microbial signatures distinguishing healthy individuals from those with OSCC. It uses QIIME 2 for microbial processing and Python-based machine learning models for classification and biomarker discovery.

📁 Repository Structure

🧬 QIIME 2 Pipeline
qiime_pipeline.sh: Shell script for running QIIME 2 pipeline locally.
manifest.tsv: Sample-to-file mapping for QIIME 2.
metadata.tsv: Sample metadata used in diversity and statistical analyses.
exported-taxonomy.tsv: Taxonomic assignment results.
feature-table.csv: Exported feature table of ASVs/OTUs used for analysis.

📊 Statistical Analysis & Machine Learning
stats_and_classification.py: Main Python script for statistical testing and supervised ML modeling.
ttest_results_with_taxa.csv: Output of statistical t-tests on taxa abundances.
model_comparison.csv: Performance metrics (AUC, F1, precision, recall) across all models.

📈 Visualizations
pca_plot.png
heatmap_top20.png
volcano_plot.png
pvalue_histogram.png
top20_features.png

🤖 Machine Learning Outputs
Each classifier has its own confusion matrix and classification report:

Random Forest:
RandomForest_confusion.png
RandomForest_feature_importance.png
RandomForest_report.txt

XGBoost:
XGBoost_confusion.png
XGBoost_feature_importance.png
XGBoost_report.txt

SVM (Sigmoid Kernel):
SVM_sigmoid_confusion.png
SVM_sigmoid_report.txt

Logistic Regression:
LogisticRegression_confusion.png
LogisticRegression_report.txt

Naïve Bayes:
NaiveBayes_confusion.png
NaiveBayes_report.txt

Ensemble Model:
Ensemble_confusion.png
Ensemble_report.txt

roc_curves.png: ROC curves comparing all models.

⚙️ Requirements
Python packages:
scikit-learn
xgboost
pandas
numpy
matplotlib
seaborn
scipy

QIIME 2 was used for microbiome processing. Most preprocessing was performed via Galaxy Europe.

📌 How to Reproduce
Run QIIME 2 Pipeline: Use qiime_pipeline.sh with your manifest and metadata files. or use Galaxy.
Run Analysis: Execute stats_and_classification.py to generate statistical tests, model results, and plots.
Review Outputs: Refer to model_comparison.csv, confusion matrices, ROC curves, and feature plots for evaluation.
