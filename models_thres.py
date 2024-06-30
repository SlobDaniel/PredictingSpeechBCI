import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats  import ttest_1samp, ttest_ind
from sklearn.ensemble           import RandomForestClassifier
from sklearn.linear_model       import LogisticRegression
from sklearn.metrics            import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection    import KFold, GridSearchCV
from sklearn.svm                import SVC

pd.set_option('display.max_rows', None)

df = pd.read_csv(r'C:\Users\edith\Thesis_nieuw\Python workfiles\building\wordprod_corr_thres.csv')
df = df.replace({'F': 0, 'M': 1}, regex=True)

##train and label data
x = df[['age', 'sex',
       'x', 'y', 'z', 'ctx_rh_S_temporal_sup_',
       'ctx_rh_G_pariet_inf-Supramar_', 'Right-Hippocampus_',
       'ctx_rh_Lat_Fis-post_', 'WM-hypointensities_',
       'ctx_rh_S_circular_insula_inf_', 'Left-Hippocampus_',
       'ctx_rh_G_temporal_middle_', 'ctx_lh_G_pariet_inf-Supramar_',
       'ctx_rh_G_front_inf-Opercular_']]

y = df['correlation']>np.median(df['correlation'])
print(np.median(df['correlation']))

# Define grind parameters for each classifier
param_grid_lr = {
    'C': [0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'penalty': ['l2']
}

param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

##building models - logistic regression, support vector machine and random forest
lr_clf = LogisticRegression(max_iter=1000)
svm_clf = SVC()
rf_clf = RandomForestClassifier(n_estimators=100, oob_score=True) #, random_state=11

# Grid search 

#grid_lr = GridSearchCV(estimator=lr_clf, param_grid=param_grid_lr, cv=5)
#grid_svm = GridSearchCV(estimator=svm_clf, param_grid=param_grid_svm, cv=10)
#grid_rf = GridSearchCV(estimator=rf_clf, param_grid=param_grid_rf, cv=10)

# Fit the grids on the training data
#grid_lr.fit(x, y)
#grid_svm.fit(x, y)
#grid_rf.fit(x, y)

# Initialize lists to store accuracy scores
accuracy_scores_lr = []
accuracy_scores_svm = []
accuracy_scores_rf = []

# initialize confusion matrixes
cm_lr = np.zeros((2, 2), dtype=int)
cm_svm = np.zeros((2, 2), dtype=int)
cm_rf = np.zeros((2, 2), dtype=int)

# Perform K-Fold cross-validation
kf = KFold(n_splits=10, shuffle=False) #, random_state=11

for train_index, test_index in kf.split(x):
    train_X, test_X = x.iloc[train_index], x.iloc[test_index]
    train_y, test_y = y.iloc[train_index], y.iloc[test_index]
    
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

    # create confusion matrices
    cm_lr += confusion_matrix(test_y, predictions_lr)
    cm_svm += confusion_matrix(test_y, predictions_svm)
    cm_rf += confusion_matrix(test_y, predictions_rf)

#collect accuracy scores in dict
accuracy_scores_dict = {
    'Logistic Regression': accuracy_scores_lr,
    'SVM': accuracy_scores_svm,
    'Random Forest': accuracy_scores_rf
}


# Print the accuracy scores for each fold, the avg accuracy, the OoB score and confusion matrices
print()
print("LR Accuracy scores for each fold:", accuracy_scores_lr)
print()
print("SVM Accuracy scores for each fold:", accuracy_scores_svm)
print()
print("RF Accuracy scores for each fold:", accuracy_scores_rf)
print()
print()
print("LR Average accuracy:", np.mean(accuracy_scores_lr))
print("SVM Average accuracy:", np.mean(accuracy_scores_svm))
print("RF Average accuracy:", np.mean(accuracy_scores_rf))
print()

#print("OOB Score:", rf_clf.oob_score)
rf_clf.fit(x, y)
oob_score = rf_clf.oob_score_
print("OOB Score:", oob_score)

# Plot confusion matrix

global_min = min(cm_lr.min(), cm_svm.min(), cm_rf.min())
global_max = max(cm_lr.max(), cm_svm.max(), cm_rf.max())

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=[0, 1])
disp_lr.plot(ax=axes[0], cmap='viridis', colorbar=False)
im = axes[0].images[0]
im.set_clim(global_min, global_max)  # Set the color limits
axes[0].set_title('Logistic Regression')

# SVM
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=[0, 1])
disp_svm.plot(ax=axes[1], cmap='viridis', colorbar=False)
im = axes[1].images[0]
im.set_clim(global_min, global_max)  # Set the color limits
axes[1].set_title('SVM')

# Random Forest
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=[0, 1])
disp_rf.plot(ax=axes[2], cmap='viridis', colorbar=False)
im = axes[2].images[0]
im.set_clim(global_min, global_max)  # Set the color limits
axes[2].set_title('Random Forest')

# Add a single colorbar for all three plots
fig.colorbar(im, ax=axes, orientation='vertical', fraction=.1)

plt.show()
# Extract feature importances
feature_importances = rf_clf.feature_importances_

# Feature importance
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
#print(feature_importance_df)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Assuming your DataFrame is named df and has 'age' and 'correlation' columns
# Example data (remove this if you already have df)
# df = pd.DataFrame({
#     'age': np.random.randint(20, 70, 100),
#     'correlation': np.random.rand(100)
# })

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

r = np.corrcoef(df['age'], df['correlation'], rowvar=False)
print(r)

# One-sample t-test
# sample mean and sd
mean_lr = np.mean(accuracy_scores_lr)
std_lr = np.std(accuracy_scores_lr)
mean_svm = np.mean(accuracy_scores_svm)
std_svm = np.std(accuracy_scores_svm)
mean_rf = np.mean(accuracy_scores_rf)
std_rf = np.std(accuracy_scores_rf)

# 'Population mean' to test against
chance_thres = 0.5

logistic_regression_thres_ttest = stats.ttest_1samp(accuracy_scores_lr, chance_thres)
svm_thres_ttest = stats.ttest_1samp(accuracy_scores_svm, chance_thres)
random_forest_thres_ttest = stats.ttest_1samp(accuracy_scores_rf, chance_thres)

print("\033[1mMeans and Standard Deviations (Binary Classification):\033[0m")
print("Logistic Regression: mean = {:.4f}, std = {:.4f}".format(mean_lr, std_lr))
print("SVM: mean = {:.4f}, std = {:.4f}".format(mean_svm, std_svm))
print("Random Forest: mean = {:.4f}, std = {:.4f}".format(mean_rf, std_rf))
print()

print("\033[1mOne-sample t-tests against chance levels (Binary Classification):\033[0m")
print("Logistic Regression: t-statistic = {:.4f}, p-value = {:.4f}".format(logistic_regression_thres_ttest.statistic, logistic_regression_thres_ttest.pvalue))
print("SVM: t-statistic = {:.4f}, p-value = {:.4f}".format(svm_thres_ttest.statistic, svm_thres_ttest.pvalue))
print("Random Forest: t-statistic = {:.4f}, p-value = {:.4f}".format(random_forest_thres_ttest.statistic, random_forest_thres_ttest.pvalue))
print()

## two-sample t-test

logreg_vs_svm_bin_ttest = stats.ttest_ind(accuracy_scores_lr, accuracy_scores_svm)
logreg_vs_rf_bin_ttest = stats.ttest_ind(accuracy_scores_lr, accuracy_scores_rf)
rf_vs_svm_bin_ttest = stats.ttest_ind(accuracy_scores_rf, accuracy_scores_svm)

print("\033[1mTwo-sample t-tests against models (Binary Classification):\033[0m")
print("Logistic Regression vs SVM: t-statistic = {:.4f}, p-value = {:.4f}".format(logreg_vs_svm_bin_ttest.statistic, logreg_vs_svm_bin_ttest.pvalue))
print("Logistic Regression vs Random Forest: t-statistic = {:.4f}, p-value = {:.4f}".format(logreg_vs_rf_bin_ttest.statistic, logreg_vs_rf_bin_ttest.pvalue))
print("Random Forest vs SVM: t-statistic = {:.4f}, p-value = {:.4f}".format(rf_vs_svm_bin_ttest.statistic, rf_vs_svm_bin_ttest.pvalue))

import matplotlib.pyplot as plt
import numpy as np

# Provided accuracy scores
accuracy_scores_bin = {
    'Logistic Regression': 0.6350,
    'SVM': 0.5093,
    'Random Forest': 0.6941
}

accuracy_scores_thres = {
    'Logistic Regression': 0.4316,
    'SVM': 0.4159,
    'Random Forest': 0.4847
}

# Prepare data for plotting
models = list(accuracy_scores_bin.keys())
x = np.arange(len(models))
width = 0.35

# Adjust the bar plot to use two shades of blue
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors for the bars
color_bin = '#1f77b4'  # A shade of blue
color_thres = '#aec7e8'  # A lighter shade of blue

# Plotting data with new colors and updated data
rects1 = ax.bar(x - width/2, [accuracy_scores_bin[model] for model in models], width, label='Binary Classification', color=color_bin)
rects2 = ax.bar(x + width/2, [accuracy_scores_thres[model] for model in models], width, label='Multiclass Classification', color=color_thres)

# Adding labels and title
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Model Performance: Binary vs. Multiclass Classification')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()