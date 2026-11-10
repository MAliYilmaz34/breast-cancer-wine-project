# Machine Learning Classification & Clustering Project

This project explores and compares multiple machine learning algorithms across two distinct domains: biomedical (Breast Cancer Wisconsin) and chemical (Wine Recognition). It covers the full machine learning pipeline from exploratory data analysis and feature scaling to model validation, test evaluation, and unsupervised clustering.

---

## 🚀 Project Pipeline & Structure

1. **Data Preprocessing & Scaling**: 
   - Handled dataset loading and cleaning.
   - Implemented `StandardScaler` strictly fitted on training sets to prevent data leakage.
   - Evaluated the impact of feature scaling on distance-sensitive algorithms.

2. **Supervised Classification**:
   - Trained and optimized **Decision Trees**, **K-Nearest Neighbors (KNN)**, and **Gaussian Naive Bayes**.
   - Tuned key hyperparameters (e.g., Tree max depth, KNN neighbor counts $k$).
   - Ensured robustness and model stability by using 3 different random seeds (`42, 52, 62`) with macro-F1 cross-validation evaluation.

3. **Test Evaluation**:
   - Selected the best configurations based solely on validation scores.
   - Evaluated final performance on a held-out test set using Confusion Matrices, Classification Reports, and ROC curves.

4. **Unsupervised Clustering (K-Means)**:
   - Applied K-Means clustering without target labels to inspect natural data groupings ($k=2$ for Breast Cancer, $k=3$ for Wine).
   - Analyzed cluster alignment using Adjusted Rand Index (ARI) and Silhouette scores.

---

## 📊 Key Findings

- **Feature Scaling Sensitivity**: Distance-based algorithms like KNN showed dramatic performance improvements after scaling (e.g., Wine KNN validation macro-F1 jumped significantly), whereas tree-based models remained completely invariant to monotonic feature scaling.
- **Hyperparameter Limits**: Increasing tree depth or neighbor counts beyond optimal thresholds led to training overfitting without validation gains.
- **Dataset Dependency**: Optimal configurations varied between datasets, proving that problem-specific validation is essential.

---

## 🛠️ Tech Stack & Requirements

- Python 3.x
- Libraries: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

To install required dependencies:
```bash
pip install -r requirements.txt
