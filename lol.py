# Importing required libraries
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv('C:\\Users\\saita\\OneDrive - UMBC\\Desktop\\733w\\red_wine.csv')

# Prepare the dataset
X = data.drop('type', axis=1)
y = data['type']

# Encode the categorical target variable (type)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Define the models with default parameters
models = {
    'Baseline (Majority class)': 'baseline',  # Baseline
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM-Linear': SVC(kernel='linear', probability=True),
    'SVM-RBF': SVC(kernel='rbf', probability=True),
    'Random Forest': RandomForestClassifier()
}

# Evaluate models using 10-fold cross-validation
results = {}

for name, model in models.items():
    if model != 'baseline':
        auc_scores = cross_val_score(model, X, y_encoded, cv=10, scoring='roc_auc').mean()
        accuracy_scores = cross_val_score(model, X, y_encoded, cv=10, scoring='accuracy').mean()
    else:
        # Baseline metrics
        majority_class = pd.Series(y_encoded).value_counts().idxmax()
        accuracy_scores = (y_encoded == majority_class).mean()
        auc_scores = 0.5  # Random classifier

    results[name] = {'AUC': auc_scores, 'Accuracy': accuracy_scores}

# Convert results to DataFrame and display
results_df = pd.DataFrame(results).T
print(results_df)