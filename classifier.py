import time
import pickle
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score


# root dir
script_dir = os.path.dirname(os.path.abspath(__file__))
person_root = f"{script_dir}/data/person_data"

# def dict
x = []
y = []

all_entries = os.listdir(person_root)
person_names = [
	entry for entry in all_entries if os.path.isdir(
		os.path.join(
			person_root,
			entry))]

names = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
for person_name in [person_names[i] for i in names]:
	temp = []
	person_name_path = os.path.join(person_root, person_name)

	person_json = f"{person_root}/{person_name}.json"

	with open(person_json, "r") as json_file:
		data_dict = json.load(json_file)

	length = len(data_dict["landmark_feature_68"])
	y.extend([person_name] * length)
	for i in range(length):
		x.extend([data_dict["landmark_feature_68"][i]])

#  +data_dict["dct_feature"][i] + data_dict["pca_feature35"][i][:1120]

# Split dataset into training and testing set
y = [s.replace('S', '') for s in y]
X_train, X_test, y_train, y_test = train_test_split(
	x, y, test_size=0.8, random_state=42)

# Feature scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM Classifier
svm_clf = SVC(gamma='scale', random_state=42, kernel='rbf')  #
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)
print('SVM Accuracy:', accuracy_score(y_test, y_pred_svm))

# Assuming y_scores is the predicted probability for the positive class
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

precision = precision_score(y_test, y_pred_svm, average='weighted')
print("Precision:", precision)

recall = recall_score(y_test, y_pred_svm, average='weighted')
print("Recall:", recall)

f1 = f1_score(y_test, y_pred_svm, average='weighted')
print("F1 Score:", f1)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred_svm)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# k-Nearest Neighbors Classifier
knn_clf = KNeighborsClassifier(n_neighbors=12, metric='euclidean', weights='uniform', n_jobs=-1)
knn_clf.fit(X_train, y_train)
y_pred_knn = knn_clf.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# Assuming y_scores is the predicted probability for the positive class
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

precision = precision_score(y_test, y_pred_knn, average='weighted')
print("Precision:", precision)

recall = recall_score(y_test, y_pred_knn, average='weighted')
print("Recall:", recall)

f1 = f1_score(y_test, y_pred_knn, average='weighted')
print("F1 Score:", f1)


# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred_knn)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


"""

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=20, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print('Random Forest Accuracy:', accuracy_score(y_test, y_pred_rf))

# Naive Bayes Classifier
nb_clf = GaussianNB(
	var_smoothing=1e-09,
	priors=None
)
nb_clf.fit(X_train, y_train)
y_pred_nb = nb_clf.predict(X_test)
print('Naive Bayes Accuracy:', accuracy_score(y_test, y_pred_nb))
"""
