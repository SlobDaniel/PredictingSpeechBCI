import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy                      import stats
from scipy.stats                import ttest_1samp, ttest_ind
from sklearn.ensemble           import RandomForestClassifier
from sklearn.linear_model       import LogisticRegression
from sklearn.metrics            import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection    import KFold
from sklearn.preprocessing      import KBinsDiscretizer
from sklearn.svm                import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
# Load the dataset
df = pd.read_csv(r'C:\Users\edith\Thesis_nieuw\Python workfiles\building\wordprod_corr_thres.csv')
df = df.replace({'F': 0, 'M': 1}, regex=True)

# Define features and target
x = df[['age', 'sex', 'x', 'y', 'z', 'ctx_rh_S_temporal_sup_', 'ctx_rh_G_pariet_inf-Supramar_', 
        'Right-Hippocampus_', 'ctx_rh_Lat_Fis-post_', 'WM-hypointensities_', 
        'ctx_rh_S_circular_insula_inf_', 'Left-Hippocampus_', 'ctx_rh_G_temporal_middle_', 
        'ctx_lh_G_pariet_inf-Supramar_', 'ctx_rh_G_front_inf-Opercular_']]

# Binning 'correlation' into three categories
y = df['correlation']
k_bins = 3  # Number of bins
est = KBinsDiscretizer(n_bins=k_bins, encode='ordinal', strategy='quantile')
y_binned = est.fit_transform(y.values.reshape(-1, 1)).astype(int).flatten()

# Initialize classifiers
lr_clf = LogisticRegression(max_iter=1000, multi_class='ovr')
svm_clf = SVC()
rf_clf = RandomForestClassifier(n_estimators=100, oob_score=True)

# Perform K-Fold cross-validation
kf = KFold(n_splits=10, shuffle=False)

# Lists to store results
accuracy_scores_lr = []
accuracy_scores_svm = []
accuracy_scores_rf = []

# Confusion matrices
cm_lr = np.zeros((k_bins, k_bins), dtype=int)
cm_svm = np.zeros((k_bins, k_bins), dtype=int)
cm_rf = np.zeros((k_bins, k_bins), dtype=int)

for train_index, test_index in kf.split(x):
    train_X, test_X = x.iloc[train_index], x.iloc[test_index]
    train_y, test_y = y_binned[train_index], y_binned[test_index]
    
    # Fit the classifiers on the training data
    lr_clf.fit(train_X, train_y)
    svm_clf.fit(train_X, train_y)
    rf_clf.fit(train_X, train_y)

    # Predict on the test data
    predictions_lr = lr_clf.predict(test_X)
    predictions_svm = svm_clf.predict(test_X)
    predictions_rf = rf_clf.predict(test_X)

    # Calculate accuracy and store it
    accuracy_lr = accuracy_score(test_y, predictions_lr)
    accuracy_svm = accuracy_score(test_y, predictions_svm)
    accuracy_rf = accuracy_score(test_y, predictions_rf)

    accuracy_scores_lr.append(accuracy_lr)
    accuracy_scores_svm.append(accuracy_svm)
    accuracy_scores_rf.append(accuracy_rf)

    # Update confusion matrices
    cm_lr += confusion_matrix(test_y, predictions_lr, labels=[0, 1, 2])
    cm_svm += confusion_matrix(test_y, predictions_svm, labels=[0, 1, 2])
    cm_rf += confusion_matrix(test_y, predictions_rf, labels=[0, 1, 2])

# Print accuracy scores for each classifier
print("LR Accuracy scores for each fold:", accuracy_scores_lr)
print("SVM Accuracy scores for each fold:", accuracy_scores_svm)
print("RF Accuracy scores for each fold:", accuracy_scores_rf)

# Print average accuracy
print("LR Average accuracy:", np.mean(accuracy_scores_lr))
print("SVM Average accuracy:", np.mean(accuracy_scores_svm))
print("RF Average accuracy:", np.mean(accuracy_scores_rf))

# Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

global_min = min(cm_lr.min(), cm_svm.min(), cm_rf.min())
global_max = max(cm_lr.max(), cm_svm.max(), cm_rf.max())

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=[0, 1, 2])
disp_lr.plot(ax=axes[0], cmap='viridis', colorbar=False)
im = axes[0].images[0]
im.set_clim(global_min, global_max)  # Set the color limits
axes[0].set_title('Logistic Regression')

# SVM
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=[0, 1, 2])
disp_svm.plot(ax=axes[1], cmap='viridis', colorbar=False)
im = axes[1].images[0]
im.set_clim(global_min, global_max)  # Set the color limits
axes[1].set_title('SVM')

# Random Forest
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=[0, 1, 2])
disp_rf.plot(ax=axes[2], cmap='viridis', colorbar=False)
im = axes[2].images[0]
im.set_clim(global_min, global_max)  # Set the color limits
axes[2].set_title('Random Forest')

# Add a single colorbar for all three plots
fig.colorbar(im, ax=axes, orientation='vertical', fraction=.1)

plt.show()

# Extract feature importances
feature_importances = rf_clf.feature_importances_

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({
    'Feature': x.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot/print feature importances
plt.figure(figsize=(12, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from Random Forest')
plt.gca().invert_yaxis()
#plt.show()

# Fit an OLS model
X = df['age']
y = df['correlation']
X = sm.add_constant(X)  # Adds a constant term to the predictor

model = sm.OLS(y, X).fit()
predictions = model.predict(X)  # Predictions from the model

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(df['age'], df['correlation'], alpha=0.5, label='Data points')
plt.plot(df['age'], predictions, color='red', label='OLS regression line')
plt.title('Scatter Plot of Age vs Correlation Score per Electrode')
plt.xlabel('Age')
plt.ylabel('Correlation Score')
plt.legend()
plt.grid(True)
plt.show()

# Print the summary of the model
print(model.summary())


# one sample t-test 
# sample mean and sd
mean_lr = np.mean(accuracy_scores_lr)
std_lr = np.std(accuracy_scores_lr)
mean_svm = np.mean(accuracy_scores_svm)
std_svm = np.std(accuracy_scores_svm)
mean_rf = np.mean(accuracy_scores_rf)
std_rf = np.std(accuracy_scores_rf)

# 'Population mean' to test against
chance_bin = 0.33

logistic_regression_bin_ttest = stats.ttest_1samp(accuracy_scores_lr, chance_bin)
svm_bin_ttest = stats.ttest_1samp(accuracy_scores_svm, chance_bin)
random_forest_bin_ttest = stats.ttest_1samp(accuracy_scores_rf, chance_bin)

print()
print("\033[1mMeans and Standard Deviations (Binary Classification):\033[0m")
print("Logistic Regression: mean = {:.4f}, std = {:.4f}".format(mean_lr, std_lr))
print("SVM: mean = {:.4f}, std = {:.4f}".format(mean_svm, std_svm))
print("Random Forest: mean = {:.4f}, std = {:.4f}".format(mean_rf, std_rf))
print()


print("\033[1mOne-sample t-tests against chance levels (Binning in three categories):\033[0m")
print("Logistic Regression: t-statistic = {:.4f}, p-value = {:.4f}".format(logistic_regression_bin_ttest.statistic, logistic_regression_bin_ttest.pvalue))
print("SVM: t-statistic = {:.4f}, p-value = {:.4f}".format(svm_bin_ttest.statistic, svm_bin_ttest.pvalue))
print("Random Forest: t-statistic = {:.4f}, p-value = {:.4f}".format(random_forest_bin_ttest.statistic, random_forest_bin_ttest.pvalue))
print()

## two sample t-test

logreg_vs_svm_bin_ttest = stats.ttest_ind(accuracy_scores_lr, accuracy_scores_svm)
logreg_vs_rf_bin_ttest = stats.ttest_ind(accuracy_scores_lr, accuracy_scores_rf)
rf_vs_svm_bin_ttest = stats.ttest_ind(accuracy_scores_rf, accuracy_scores_svm)

print("\033[1mTwo-sample t-tests against models (Binning in three categories):\033[0m")
print("Logistic Regression vs SVM: t-statistic = {:.4f}, p-value = {:.4f}".format(logreg_vs_svm_bin_ttest.statistic, logreg_vs_svm_bin_ttest.pvalue))
print("Logistic Regression vs Random Forest: t-statistic = {:.4f}, p-value = {:.4f}".format(logreg_vs_rf_bin_ttest.statistic, logreg_vs_rf_bin_ttest.pvalue))
print("Random Forest vs SVM: t-statistic = {:.4f}, p-value = {:.4f}".format(rf_vs_svm_bin_ttest.statistic, rf_vs_svm_bin_ttest.pvalue))