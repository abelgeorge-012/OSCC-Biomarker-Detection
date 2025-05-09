# Core libraries
import os
import numpy as np
import pandas as pd

# Stats and math
from scipy.stats import ttest_ind

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing and dimensionality reduction
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

# Model selection and evaluation
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Machine learning models
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

# Oversampling
from imblearn.over_sampling import SMOTE

# Model performance
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ----------------- TAXONOMY UTILS -----------------

def load_taxonomy_map(taxonomy_path):
    taxonomy = pd.read_csv(taxonomy_path, sep='\t')
    # Map Feature ID to short taxon name (after last ';')
    def short_taxon(taxon):
        if pd.isna(taxon):
            return ""
        return taxon.split(';')[-1].strip()
    return {row['Feature ID']: short_taxon(row['Taxon']) for _, row in taxonomy.iterrows()}

def map_features_to_taxa(features, taxonomy_map):
    # Use short taxon name if present, else fallback to OTU ID
    return [taxonomy_map.get(f, f) for f in features]

# ----------------- DATA LOADING -----------------

def load_and_align(abundance_path, metadata_path):
    abundance = pd.read_csv(abundance_path, index_col=0)
    metadata = pd.read_csv(metadata_path, sep='\t', index_col='sample id')
    common = abundance.columns.intersection(metadata.index)
    abundance = abundance[common].T
    metadata = metadata.loc[common]
    return abundance, metadata

# ----------------- T-TESTS AND PLOTS -----------------

def perform_ttests(abundance, metadata, output_dir, taxonomy_map):
    os.makedirs(output_dir, exist_ok=True)

    # Normalize column names
    metadata.columns = metadata.columns.str.strip()

    # Use the correct column name directly
    mask_healthy = metadata['Condition'].str.lower() == 'healthy'
    mask_disease = metadata['Condition'].str.lower() != 'healthy'
    healthy_samples = abundance[mask_healthy]
    disease_samples = abundance[mask_disease]

    # Run t-tests
    results = []
    for feature in abundance.columns:
        t_stat, p_val = ttest_ind(
            healthy_samples[feature],
            disease_samples[feature],
            equal_var=False
        )
        results.append({'Feature': feature, 'p_value': p_val})

    ttest_df = pd.DataFrame(results)

    # Compute log2 fold change
    log2fc = []
    for feature in ttest_df['Feature']:
        mean_h = healthy_samples[feature].mean()
        mean_d = disease_samples[feature].mean()
        fc = (mean_d + 1e-6) / (mean_h + 1e-6)
        log2fc.append(np.log2(fc))
    ttest_df['log2fc'] = log2fc
    ttest_df['-log10(p)'] = -np.log10(ttest_df['p_value'])

    # Add short taxon column
    ttest_df['Taxon'] = map_features_to_taxa(ttest_df['Feature'], taxonomy_map)

    # Save CSV
    ttest_df.sort_values('p_value').to_csv(os.path.join(output_dir, 'ttest_results_with_taxa.csv'), index=False)

    # Volcano Plot
    plt.figure(figsize=(10,6))
    plt.scatter(ttest_df['log2fc'], ttest_df['-log10(p)'], alpha=0.7)
    plt.axhline(-np.log10(0.05), color='red', linestyle='--', label='p = 0.05')
    plt.xlabel('log2(Fold Change: Disease / Healthy)')
    plt.ylabel('-log10(p-value)')
    plt.title('Volcano Plot of t-test Results')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'volcano_plot.png'))
    plt.close()

    # Histogram of p-values
    plt.figure(figsize=(8,5))
    plt.hist(ttest_df['p_value'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of p-values')
    plt.xlabel('p-value')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'pvalue_histogram.png'))
    plt.close()

    # Top 20 significant features bar plot (use short taxon names)
    top20 = ttest_df.sort_values('p_value').head(20)
    plt.figure(figsize=(10,6))
    plt.barh(top20['Taxon'], -np.log10(top20['p_value']), color='orange')
    plt.xlabel('-log10(p-value)')
    plt.title('Top 20 Significant Bacteria (t-test)')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top20_features.png'))
    plt.close()

    # PCA plot
    scaler = StandardScaler()
    abundance_scaled = scaler.fit_transform(abundance)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(abundance_scaled)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=abundance.index)
    pca_df['Condition'] = metadata['Condition'].values

    plt.figure(figsize=(8,6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Condition', s=80)
    plt.title('PCA of Microbiome Samples')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_plot.png'))
    plt.close()

    # Heatmap of top 20 features (use short taxon names)
    top20_features = top20['Feature'].tolist()
    heatmap_data = abundance[top20_features]
    heatmap_data = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()  # z-score normalization
    row_colors = metadata['Condition'].map({'healthy': 'skyblue', 'oscc': 'salmon'})

    # Rename columns to short taxon names for the heatmap
    heatmap_data.columns = map_features_to_taxa(heatmap_data.columns, taxonomy_map)

    sns.clustermap(heatmap_data, cmap='vlag', row_colors=row_colors, figsize=(12, 10))
    plt.suptitle('Heatmap of Top 20 Differential Bacteria', y=1.02)
    plt.savefig(os.path.join(output_dir, 'heatmap_top20.png'))
    plt.close()

    print("T-test analysis and visualizations saved to:", output_dir)

# ----------------- CLASSIFICATION AND FEATURE IMPORTANCE -----------------

def classify_and_tune_models(abundance, metadata, output_dir, taxonomy_map):
    os.makedirs(output_dir, exist_ok=True)
    # Log transform and feature filtering
    abundance = np.log1p(abundance)
    # Remove zero/low variance features
    vt = VarianceThreshold(threshold=0.0)
    abundance = pd.DataFrame(vt.fit_transform(abundance),
                             index=abundance.index,
                             columns=[abundance.columns[i] for i in vt.get_support(indices=True)])

    # Labels
    le = LabelEncoder()
    y = le.fit_transform(metadata['Condition'].str.lower())
    X = abundance.values
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)

    # Oversample minority
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # Define models and hyperparameter spaces
    models = {
        'RandomForest': (RandomForestClassifier(random_state=42), {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [None, 10, 20, 30, 50, 100],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8]
        }),
        'XGBoost': (xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), {
            'n_estimators': [100, 200, 300, 500, 800],
            'max_depth': [3, 6, 10, 15],
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }),
        'SVM_sigmoid': (
            SVC(kernel='sigmoid', probability=True, random_state=42),
            {
                'C': [0.01, 0.1, 1, 10, 100],
                'gamma': ['scale', 'auto'],
                'coef0': [0.0, 0.1, 0.5, 1.0],
                'class_weight': ['balanced', None]
        }),
        'LogisticRegression': (LogisticRegression(max_iter=1000, random_state=42), {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        }),
        'NaiveBayes': (GaussianNB(), {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        })
    }

    best_models = {}
    for name, (model, params) in models.items():
        print(f"Tuning {name}...")
        rs = RandomizedSearchCV(model, params, n_iter=100, cv=10,
                                scoring='accuracy', random_state=42, n_jobs=-1)
        rs.fit(X_res, y_res)
        best_models[name] = rs.best_estimator_
        print(f"Best {name} params: {rs.best_params_}")

    # Optionally add a voting ensemble
    ensemble = VotingClassifier(
        estimators=[(n, m) for n, m in best_models.items()], voting='soft')
    ensemble.fit(X_res, y_res)
    best_models['Ensemble'] = ensemble

    # Evaluate
    summary = []
    for name, model in best_models.items():
        preds = model.predict(X_test)
        summary.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds),
            'Recall': recall_score(y_test, preds),
            'F1-score': f1_score(y_test, preds)
        })
        # Save report and confusion matrix
        rep = classification_report(y_test, preds, target_names=le.classes_)
        with open(os.path.join(output_dir, f'{name}_report.txt'), 'w') as f:
            f.write(rep)
        cm = confusion_matrix(y_test, preds)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.colorbar()
        ticks = np.arange(len(le.classes_))
        plt.xticks(ticks, le.classes_, rotation=45)
        plt.yticks(ticks, le.classes_)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_confusion.png'))
        plt.close()

    pd.DataFrame(summary).to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    print(f"Model tuning and evaluation results saved to {output_dir}")

    feature_names = abundance.columns

    # Plot for each model that supports feature importance (use short taxon names)
    for model_name in ['RandomForest', 'XGBoost']:
        if model_name in best_models:
            plot_feature_importance(best_models[model_name], feature_names, model_name, output_dir, taxonomy_map)

    print(f"Importance features saved to {output_dir}")

    plot_roc_curves(best_models, X_test, y_test, output_dir, le.classes_)
    print(f"ROC AOC curve saved to {output_dir}")

def plot_roc_curves(models, X_test, y_test, output_dir, class_names):
    plt.figure(figsize=(10, 7))
    # Binarize labels if needed
    y_test_bin = label_binarize(y_test, classes=[0, 1]).ravel()
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            probs = model.decision_function(X_test)
        else:
            continue  # Skip models without probability or decision scores
        fpr, tpr, _ = roc_curve(y_test_bin, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()

def plot_feature_importance(model, feature_names, model_name, output_dir, taxonomy_map, top_n=20):
    """Plot top N feature importances for tree-based models, using short taxon names."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    # Map OTU IDs to short taxon names
    top_labels = map_features_to_taxa(top_features, taxonomy_map)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_labels, palette='viridis')
    plt.title(f'Top {top_n} Important Bacteria - {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Bacteria (Taxon)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_feature_importance.png'))
    plt.close()

# ----------------- MAIN SCRIPT -----------------

if __name__ == "__main__":
    # Set file paths
    abundance_file = 'feature-table.csv'
    taxonomy_file = 'exported-taxonomy.tsv'
    metadata_file = 'metadata.tsv'
    output_dir = 'results'

    # Check if files exist
    if not os.path.exists(abundance_file):
        raise FileNotFoundError(f"Abundance file not found: {abundance_file}")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    if not os.path.exists(taxonomy_file):
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_file}")

    # Load taxonomy map (short names)
    taxonomy_map = load_taxonomy_map(taxonomy_file)

    # Run analysis
    abundance, metadata = load_and_align(abundance_file, metadata_file)
    perform_ttests(abundance, metadata, output_dir, taxonomy_map)
    classify_and_tune_models(abundance, metadata, output_dir, taxonomy_map)

    print(f"All results are saved in the '{output_dir}' folder.")
