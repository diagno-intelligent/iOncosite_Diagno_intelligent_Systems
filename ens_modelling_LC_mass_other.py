import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.svm import SVC
from scipy import stats
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
# Load data
#"E:/project_new/LC_NonLC/Ensemble_model/selected_models/scaler_ALL_FEATURE_LC_mass_other_rf_chi2_BOTH__min_max_w_fec.pkl"

file_path = "E:/project_new/LC_NonLC/Features_ML_model_inc/inc_V3_20d_8b_LC_mass_others_features_whole_combined.csv"
#file_path = "D:/clavicle_new_mes_reg_score3/Female_3pt_1pt_new_measurement _whole.xlsx"
data1  = pd.read_csv(file_path)
X = data1.iloc[:17000,2:]   #independent columns
print(X)
y = data1.iloc[:17000,1]    #target column i.e w_m_cl_b WITHOUT_TRIM cl_b M_cl_b
print(y)


X1=X
y1=y

list_id = data1.iloc[:17000, 0]
list_ID = data1.iloc[:17000, 0]
import pickle
import os
print("\n rf_chi2")
def scale_datasets(X, scaler_path):
    #print(f"Attempting to open scaler file at: {repr(scaler_path)}")
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler.transform(X)

# Define the path to the scaler file
scaler_path = r"E:/project_new/LC_NonLC/Ensemble_model/selected_models/scaler_ALL_FEATURE_LC_mass_other_rf_chi2_BOTH__min_max_w_fec.pkl"

# Print the path to verify
#print(f"Scaler path: {repr(scaler_path)}")
X_test1 = pd.DataFrame(scale_datasets(X1, scaler_path))

#X_test1 = scale_datasets(X1)

from sklearn.linear_model import Lasso

    #X_train_selected = rfe.transform(X_train)
with open(f"E:/project_new/LC_NonLC/Ensemble_model/selected_models/selected_features_LC_mass_others_rf_chi2_BOTH_w_fec_200.txt", 'r') as file:
    selected_feature_indices = list(map(int, file.read().split(',')))

X_test_selected1 = X_test1.iloc[:, selected_feature_indices]

selected_feature_names = X1.columns[selected_feature_indices]
#print("Selected features:", selected_feature_names)

loaded_SVM_model = joblib.load(f"E:/project_new/LC_NonLC/Ensemble_model/selected_models/lbm_BOTH_rf_model_chi2_w_fec_200_train_acc1.0_test_acc0.914235294117647.pkl")


y_pred1=rf_chi2_LC_NR = loaded_SVM_model.predict(X_test_selected1)
print('y_pred1',y_pred1)
unique_labels = np.unique(y_pred1)
print("Unique Labels:", unique_labels)
for i in range (0,len(y_pred1)):
    #print('y_pred1[i]',y_pred1[i])
    if y_pred1[i]==1:
        y_pred1[i]=1
    if y_pred1[i]==0:
        y_pred1[i]=0
rf_chi2_LC_NR=y_pred1
test_accuracy1 = accuracy_score(y1, y_pred1)
#auc_test1 = roc_auc_score(y1, y_pred1)
#f1_test1 = f1_score(y1, y_pred1)
print("rf_chi2 test Accuracy on whole data:", test_accuracy1)

cm_test1 = confusion_matrix(y1, y_pred1)
cm_test_df1 = pd.DataFrame(cm_test1, index=['Lung cancer_mass','others'], columns=['Lung cancer_mass','others'])
plt.figure(figsize=(5, 4))
sns.heatmap(cm_test_df1, annot=True)
plt.title(f"whole_Confusion Matrix_LC_NR_rf_mi_Test")
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')

class_wise_accuracy1 = np.diag(cm_test1) / cm_test1.sum(axis=1)
print("\nWhole data Class-wise accuracy:")
total_acc = 0
for i, acc in enumerate(class_wise_accuracy1):
    print(f"Class {i}: {acc:.2f}")
    total_acc += acc

total_accuracy1 = total_acc / 2
print(f"Whole case accuracy: {total_accuracy1:.2f}")
rf_chi2_LC_NR = rf_chi2_LC_NR.reshape(-1, 1)
rf_mi_5m1=[]
for i in range (0,len(rf_chi2_LC_NR)):
    rf_mi_5m1.append(rf_chi2_LC_NR[i,0])
rf_chi2_LC_NR=rf_mi_5m1
#############33
print("\n xgb_chi2")
def scale_datasets(X, scaler_path):
    #print(f"Attempting to open scaler file at: {repr(scaler_path)}")
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler.transform(X)

# Define the path to the scaler file
scaler_path = r"E:/project_new/LC_NonLC/Ensemble_model/selected_models/scaler_ALL_FEATURE_2_LC_mass_other_xgb_chi2__min_max_K_{k}.pkl"

# Print the path to verify
#print(f"Scaler path: {repr(scaler_path)}")
X_test1 = pd.DataFrame(scale_datasets(X1, scaler_path))

#X_test1 = scale_datasets(X1)

from sklearn.linear_model import Lasso

    #X_train_selected = rfe.transform(X_train)
with open(f"E:/project_new/LC_NonLC/Ensemble_model/selected_models/selected_features_2_LC_mass_other_xgb_chi2_k150.txt", 'r') as file:
    selected_feature_indices = list(map(int, file.read().split(',')))

X_test_selected1 = X_test1.iloc[:, selected_feature_indices]

selected_feature_names = X1.columns[selected_feature_indices]
#print("Selected features:", selected_feature_names)

loaded_SVM_model = joblib.load(f"E:/project_new/LC_NonLC/Ensemble_model/selected_models/2_LC_mass_other_xgb_chi2_fec_150_acc1.0.pkl")


y_pred1=xgb_chi2_LC_NR = loaded_SVM_model.predict(X_test_selected1)
print(y_pred1)
for i in range (0,len(y_pred1)):
    if y_pred1[i]==1:
        y_pred1[i]=1 
    if y_pred1[i]==0:
        y_pred1[i]=0
xgb_chi2_LC_NR=y_pred1 
test_accuracy1 = accuracy_score(y1, y_pred1)
#auc_test1 = roc_auc_score(y1, y_pred1)
#f1_test1 = f1_score(y1, y_pred1)
print("svm_chi22 test Accuracy on whole data:", test_accuracy1)

cm_test1 = confusion_matrix(y1, y_pred1)
cm_test_df1 = pd.DataFrame(cm_test1, index=['Lung cancer_mass','others'], columns=['Lung cancer_mass','others'])
plt.figure(figsize=(5, 4))
sns.heatmap(cm_test_df1, annot=True)
plt.title(f"whole_Confusion Matrix_LC_NR_svm_chi2_Test")
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')

class_wise_accuracy1 = np.diag(cm_test1) / cm_test1.sum(axis=1)
print("\nWhole data Class-wise accuracy:")
total_acc = 0
for i, acc in enumerate(class_wise_accuracy1):
    print(f"Class {i}: {acc:.2f}")
    total_acc += acc

total_accuracy1 = total_acc / 2
print(f"Whole case accuracy: {total_accuracy1:.2f}")
xgb_chi2_LC_NR = xgb_chi2_LC_NR.reshape(-1, 1)
xgb_chi2_5m1=[]
for i in range (0,len(xgb_chi2_LC_NR)):
    xgb_chi2_5m1.append(xgb_chi2_LC_NR[i,0])
xgb_chi2_LC_NR=xgb_chi2_5m1

#################

print("\n xgb_annova")
def scale_datasets(X, scaler_path):
    #print(f"Attempting to open scaler file at: {repr(scaler_path)}")
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler.transform(X)

# Define the path to the scaler file
scaler_path = r"E:/project_new/LC_NonLC/Ensemble_model/selected_models/scaler_ALL_FEATURE_LC_mass_other_rf_mutual_info_classif_BOTH__min_max_w_fec.pkl"

# Print the path to verify
#print(f"Scaler path: {repr(scaler_path)}")
X_test1 = pd.DataFrame(scale_datasets(X1, scaler_path))

#X_test1 = scale_datasets(X1)

from sklearn.linear_model import Lasso

    #X_train_selected = rfe.transform(X_train)
with open(f"E:/project_new/LC_NonLC/Ensemble_model/selected_models/selected_features_LC_mass_others_rf_mutual_info_classif_BOTH_w_fec_150.txt", 'r') as file:
    selected_feature_indices = list(map(int, file.read().split(',')))

X_test_selected1 = X_test1.iloc[:, selected_feature_indices]

selected_feature_names = X1.columns[selected_feature_indices]
#print("Selected features:", selected_feature_names)

loaded_SVM_model = joblib.load(f"E:/project_new/LC_NonLC/Ensemble_model/selected_models/lbm_BOTH_rf_model_mutual_info_classif_w_fec_150_train_acc1.0_test_acc0.914235294117647.pkl")


y_pred1=rf_mi_LC_NR = loaded_SVM_model.predict(X_test_selected1)
print(y_pred1)
for i in range (0,len(y_pred1)):
    if y_pred1[i]==1:
        y_pred1[i]=1
    if y_pred1[i]==0:
        y_pred1[i]=0
rf_mi_LC_NR=y_pred1
test_accuracy1 = accuracy_score(y1, y_pred1)
#auc_test1 = roc_auc_score(y1, y_pred1)
#f1_test1 = f1_score(y1, y_pred1)
print("svm_LASSO2 test Accuracy on whole data:", test_accuracy1)

cm_test1 = confusion_matrix(y1, y_pred1)
cm_test_df1 = pd.DataFrame(cm_test1, index=['Lung cancer_mass','others'], columns=['Lung cancer_mass','others'])
plt.figure(figsize=(5, 4))
sns.heatmap(cm_test_df1, annot=True)
plt.title(f"whole_Confusion Matrix_LC_NR_Test")
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')

class_wise_accuracy1 = np.diag(cm_test1) / cm_test1.sum(axis=1)
print("\nWhole data Class-wise accuracy:")
total_acc = 0
for i, acc in enumerate(class_wise_accuracy1):
    print(f"Class {i}: {acc:.2f}")
    total_acc += acc

total_accuracy1 = total_acc / 2
print(f"Whole case accuracy: {total_accuracy1:.2f}")
rf_mi_LC_NR = rf_mi_LC_NR.reshape(-1, 1)
rf_mi_5m1=[]
for i in range (0,len(rf_mi_LC_NR)):
    rf_mi_5m1.append(rf_mi_LC_NR[i,0])
rf_mi_LC_NR=rf_mi_5m1



###############3
#########################33
#print('rf_chi2_LC_NR',rf_chi2_LC_NR)
#print('list_ID',list_id)
avg_ens=[]
for i in range (0,len(rf_chi2_LC_NR)):
    if rf_chi2_LC_NR[i]+xgb_chi2_LC_NR[i]+rf_mi_LC_NR[i]>1:
        avg_ens.append(1)
    else:
        avg_ens.append(0)
        


############### ens_STACK

import numpy as np
import joblib
import pandas as pd
import xlrd
import openpyxl
import os
from os import listdir
from os.path import isfile, join

### insert the name of the column as a string in brackets
list1 = rf_chi2_LC_NR
list2 = xgb_chi2_LC_NR
list3 = rf_mi_LC_NR 

list6 = data1.iloc[:17000,1]
print('/n')
print('stacked_ML_ML_LCmass and others')


preds_model1=list1
preds_model2=list2
preds_model3=list3 

# Combine predictions into a feature matrix
X_stack = np.column_stack((preds_model1, preds_model2, preds_model3))
y_true = list6 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_stack, y_true, test_size=0.2, random_state=42)

### Initialize the meta-classifier (final estimator)
##meta_clf = LogisticRegression()
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
# Base classifiers
rf = RandomForestClassifier(random_state=42)
et = ExtraTreesClassifier(random_state=42)
gbm = GradientBoostingClassifier(random_state=42)
ada = AdaBoostClassifier(random_state=42)

# Stacked ensemble - using Logistic Regression as meta-classifier
base_classifiers = [('Random Forest', rf), ('Extra Trees', et), ('Gradient Boosting', gbm), ('AdaBoost', ada)]
stacked_predictions_train = []

for name, clf in base_classifiers:
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    stacked_predictions_train.append(y_pred_train)

X_train_stack = np.column_stack(stacked_predictions_train)

meta_clf_lr = AdaBoostClassifier(random_state=42)
meta_clf_lr.fit(X_train_stack, y_train)

# Evaluate on test set
stacked_predictions_test = []

for name, clf in base_classifiers:
    y_pred_test = clf.predict(X_test)
    stacked_predictions_test.append(y_pred_test)

X_test_stack = np.column_stack(stacked_predictions_test)

y_pred_meta_lr = meta_clf_lr.predict(X_test_stack)
accuracy_lr = accuracy_score(y_test, y_pred_meta_lr)
#print(f"Accuracy of Logistic Regression on stacked predictions: {accuracy_lr:.4f}")
# Initialize the meta-classifier (final estimator)
#meta_clf = VotingClassifier(estimators=base_classifiers, voting='hard')
meta_clf =LogisticRegression()
# Train the meta-classifier using the stacked predictions as features
meta_clf.fit(X_train, y_train)
stacked_preds = meta_clf.predict(X_train)
accuracy = accuracy_score(y_train, stacked_preds)
print(f"TRAINING Accuracy of Logistic Regression on stacked predictions: {accuracy_lr:.4f}")
# Make predictions using the meta-classifier
stacked_preds = meta_clf.predict(X_test)

# Evaluate the performance of the stacked ensemble
accuracy = accuracy_score(y_test, stacked_preds)
print(f"Testing Accuracy of Stacked Ensemble: {accuracy}")
# Save the trained stacked ensemble model to a file
joblib.dump(meta_clf, 'stacked_ensemble_model_ML_LCmass_others.pkl')

# Load the saved stacked ensemble model from the file
loaded_model = joblib.load('stacked_ensemble_model_ML_LCmass_others.pkl')

# Use the loaded model for prediction on the single test value
y_test_p=y_test
predicted_value=predicted_value_p = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, predicted_value)
print(f"Testing Accuracy of LOADED Stacked Ensemble: {accuracy}")
# Display the predicted value
#print("Predicted Value:", predicted_value)
loaded_model = joblib.load('stacked_ensemble_model_ML_LCmass_others.pkl')
predicted_value1 = loaded_model.predict(X_stack)
#print('st_predicted_value',predicted_value)
# Compute confusion matrix

cm = confusion_matrix(y_true, predicted_value1)

# Compute class-wise accuracy
class_wise_accuracy = np.diag(cm) / cm.sum(axis=1)
for i, acc in enumerate(class_wise_accuracy):
    print(f"Class {i}: {acc:.2f}")
accuracy1 = accuracy = accuracy_score(y_true, predicted_value1)
print(f"Testing Accuracy (wholedataset)of LOADED Stacked Ensemble 5m: {accuracy1}")

accuracy = accuracy_score(y_true, avg_ens)
#print(f"Testing Accuracy (wholedataset)of AVGERAGE Ensemble 5m: {accuracy}")
SCORE3_LIST = data1.iloc[:17000, 1]
lbm_ens_res = pd.DataFrame({
    "list_ID": list_id,
    "rf_chi2_LC_NR": rf_chi2_LC_NR,
    "xgb_chi2_LC_NR":xgb_chi2_LC_NR,
    "rf_mi_LC_NR":rf_mi_LC_NR,
    
    "avg_ens":avg_ens,
    "st_ens_predicted_value":predicted_value1,
    "SCORE3_LIST ":SCORE3_LIST 
    })

lbm_ens_res.to_csv(f"class_LC_NR_stack_and_avg_result_test.csv", index=False)
print("\n Lung_cancer_MASS  and Normal class")
#print("\n OS VS NOS")
############  ST_ENS
cm_test1 = confusion_matrix(y1, predicted_value1)
cm_test_df1 = pd.DataFrame(cm_test1, index=['Lung cancer_mass','others'], columns=['Lung cancer_mass','others'])
plt.figure(figsize=(5, 4))
sns.heatmap(cm_test_df1, annot=True)
plt.title(f"ST_ENS whole_Confusion Matrix_2 class_LC_NR_Test")
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')

class_wise_accuracy1 = np.diag(cm_test1) / cm_test1.sum(axis=1)
print("\nST_ENS Whole data Class-wise accuracy:")
total_acc = 0
for i, acc in enumerate(class_wise_accuracy1):
    print(f"Class {i}: {acc*100:.2f}%")
    total_acc += acc

total_accuracy1 = total_acc / 2
print(f"ST_ENS Whole case accuracy: {total_accuracy1*100:.2f}%")
############  AVG_ENS

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm_test1 = confusion_matrix(y1, avg_ens)
cm_test_df1 = pd.DataFrame(cm_test1, index=['Lung cancer_mass','others'], columns=['Lung cancer_mass','others'])

# Plot confusion matrix heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(cm_test_df1, annot=True, fmt='d', cmap='Blues')
plt.title("AVG_ENS whole_Confusion Matrix_2 class_LC_NR_Test")
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.tight_layout()
plt.show()

# Calculate class-wise accuracy safely
with np.errstate(divide='ignore', invalid='ignore'):
    class_wise_accuracy1 = np.diag(cm_test1) / cm_test1.sum(axis=1)
    class_wise_accuracy1 = np.nan_to_num(class_wise_accuracy1)  # Replace NaN with 0

# Print class-wise accuracy
print("\nAVG_ENS Whole data Class-wise accuracy:")
class_labels = ['Lung cancer_mass','others']
total_acc = 0

for i, acc in enumerate(class_wise_accuracy1):
    print(f"{class_labels[i]}: {acc * 100:.2f}%")
    total_acc += acc

# Average accuracy
total_accuracy1 = total_acc / len(class_wise_accuracy1)
print(f"\nAVG_ENS Overall average accuracy: {total_accuracy1 * 100:.2f}%")


##############################3

from sklearn.metrics import confusion_matrix, classification_report, f1_score
import numpy as np

# Replace with your lists
y_true = y_test_p # actual labels
y_pred = predicted_value_p # predicted labels

# Confusion matrix
labels = [0, 1]  # assuming 5 classes
cm = confusion_matrix(y_true, y_pred, labels=labels)
print("Confusion Matrix:\n", cm)

# Sensitivity (Recall) = TP / (TP + FN)
sensitivity_per_class = np.diag(cm) / np.sum(cm, axis=1)
print("Sensitivity (per class):", sensitivity_per_class)

# Specificity = TN / (TN + FP)
specificity_per_class = []
for i in range(len(labels)):
    TP = cm[i, i]
    FN = np.sum(cm[i, :]) - TP
    FP = np.sum(cm[:, i]) - TP
    TN = np.sum(cm) - (TP + FP + FN)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    specificity_per_class.append(specificity)
print("Specificity (per class):", specificity_per_class)

# Overall Sensitivity = sum(TP) / sum(TP + FN)
TP_total = np.diag(cm).sum()
FN_total = np.sum(cm, axis=1).sum() - TP_total
overall_sensitivity = TP_total / (TP_total + FN_total)
print("Overall Sensitivity:", round(overall_sensitivity, 4))

# Overall Specificity = sum(TN) / sum(TN + FP)
overall_specificity = 0
for i in range(len(labels)):
    TP = cm[i, i]
    FN = np.sum(cm[i, :]) - TP
    FP = np.sum(cm[:, i]) - TP
    TN = np.sum(cm) - (TP + FP + FN)
    overall_specificity += TN
overall_specificity /= (overall_specificity + sum(np.sum(cm, axis=0) - np.diag(cm)))
print("Overall Specificity:", round(overall_specificity, 4))

# Macro F1-score
macro_f1 = f1_score(y_true, y_pred, average='macro')
print("Macro F1-score:", round(macro_f1, 4))

# Optional: Full classification report
print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))


