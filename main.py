# Improvements
# - Create dataset from scratch
# -
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

# Models used in this activity
models = [GaussianNB(), LogisticRegression()]

# Function for fitting model and calculating metrics
# global variables used as parameters. Improvement?
def create_model(model_type, X_train, y_train, X_test, y_test):
    model_name = type(model_type).__name__
    time_start = time.perf_counter()
    model = model_type.fit(X_train, y_train)
    time_elapsed = (time.perf_counter() - time_start)

    prob = model_type.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100
    auc = roc_auc_score(y_test, prob)
    fpr, tpr, _ = roc_curve(y_test, prob)

    print(f"{model_name} Metrics")
    print("Computation Time:%5.4f seconds" % time_elapsed)
    print("Accuracy: %.2f" % accuracy)
    print("Precision: %.2f" % precision)
    print("Recall: %.2f" % recall)
    print('AUC=%.3f' % auc, "\n")

    plt.plot(fpr, tpr, marker='.', label=f'{model_name}')


for model in models:
    create_model(model, X_train, y_train, X_test, y_test)

plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
