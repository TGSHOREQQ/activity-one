from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import time

SAMPLES = 100000
NO_CLASSES = 2
TEST_SIZE = 0.3
CLASS_SEP = 1

# Generating classification dataset using sklearn function
X, y = make_classification(n_samples=SAMPLES, n_classes=NO_CLASSES, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, random_state=0, class_sep=CLASS_SEP)
# Subset data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=2)

# Gaussian Naive Bayes
gnb = GaussianNB()
time_start_gnb = time.perf_counter()
gnb_model = gnb.fit(X_train, y_train)
time_elapsed_gnb = (time.perf_counter() - time_start_gnb)
# Model Values
prob_gnb = gnb.predict_proba(X_test)[:, 1]
y_pred_gnb = gnb_model.predict(X_test)
gnb_accuracy = accuracy_score(y_test, y_pred_gnb)*100
gnb_precision = precision_score(y_test, y_pred_gnb)*100
gnb_recall = recall_score(y_test, y_pred_gnb)*100

# Logistic Regression
lr = LogisticRegression()
time_start_lr = time.perf_counter()
lr_model = lr.fit(X_train, y_train)
time_elapsed_lr = (time.perf_counter() - time_start_lr)
# Model Values
prob_lr = lr.predict_proba(X_test)[:, 1]
y_pred_lr = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr)*100
lr_precision = precision_score(y_test, y_pred_lr)*100
lr_recall = recall_score(y_test, y_pred_lr)*100

# AUC
auc_gnb = roc_auc_score(y_test, prob_gnb)
auc_lr = roc_auc_score(y_test, prob_lr)

# Gaussian Naive BAyes Metrics
print("Gaussian Naive Bayes Metrics")
print("GNB Computation Time:%5.4f seconds" % time_elapsed_gnb)
print("GNB Accuracy: %.2f" % gnb_accuracy)
print("GNB Precision: %.2f" % gnb_precision)
print("GNB Recall: %.2f" % gnb_recall, "\n")
print('GNB: AUC=%.3f' % auc_gnb)

# Logistic Regression Metrics
print("Logistic Regression Metrics")
print("LR Computation Time:%5.4f seconds" % time_elapsed_lr)
print("LR Accuracy: %.2f" % lr_accuracy)
print("LR Precision: %.2f" % lr_precision)
print("LR Recall: %.2f" % lr_recall, "\n")
print('Logistic Regression: AUC=%.3f' % auc_lr)

# Calculate ROC Curves
gnb_fpr, gnb_tpr, _ = roc_curve(y_test, prob_gnb)
lr_fpr, lr_tpr, _ = roc_curve(y_test, prob_lr)

plt.plot([0, 1], [0, 1], 'r--')
plt.plot(gnb_fpr, gnb_tpr, marker='.', label='Naive Bayes')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()