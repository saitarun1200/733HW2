
# Import required libraries clearly
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load white-wine dataset clearly
white_data = pd.read_csv(r'C:\Users\saita\OneDrive - UMBC\Desktop\733w\white_wine.csv')

# Prepare dataset
X_white = white_data.drop('type', axis=1)
y_white = LabelEncoder().fit_transform(white_data['type'])

# Split the white-wine dataset clearly into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(
    X_white, y_white, test_size=0.3, random_state=42
)

# Train Naive Bayes classifier clearly on white-wine training data
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predict probabilities on test set clearly
y_probs = nb_model.predict_proba(X_test)[:, 1]

# Calculate and display AUC score clearly
white_auc = roc_auc_score(y_test, y_probs)
print(f"Naive Bayes AUC score on White Wine dataset: {white_auc:.4f}")
