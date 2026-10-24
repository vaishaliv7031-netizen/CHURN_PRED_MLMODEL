import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, classification_report, roc_auc_score, roc_curve

# 1 Load Data
df = pd.read_csv('project/data/Churn_Modelling.csv')

# 2 Preprocessing
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

#  Features & Target Split
x = df.drop('Exited', axis=1)
y = df['Exited']


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# 4. Scale Data Properly (To be used for Logistic Regression and Gradient Boosting)
scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(xtrain)
xtest_scaled = scaler.transform(xtest) # FIXED: Use transform, not fit_transform


rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'
)
rf_model.fit(xtrain, ytrain)
rf_pred = rf_model.predict(xtest)
rf_prob = rf_model.predict_proba(xtest)[:, 1]

print("Test Accuracy:", rf_model.score(xtest, ytest))
print("ROC AUC Score:", roc_auc_score(ytest, rf_prob))
print(classification_report(ytest, rf_pred))

plt.figure(figsize=(10, 5))
fpr, tpr, _ = roc_curve(ytest, rf_prob)
plt.plot(fpr, tpr, label='Random Forest', color='blue')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()



print("LOGISTIC REGRESSION ")

lr_model = LogisticRegression(class_weight='balanced', max_iter=2000)
lr_model.fit(xtrain_scaled, ytrain)
lr_prob = lr_model.predict_proba(xtest_scaled)[:, 1]
lr_pred = (lr_prob > 0.3).astype(int) 

print("Confusion Matrix:\n", confusion_matrix(ytest, lr_pred))
print("Recall:", recall_score(ytest, lr_pred))
plt.figure(figsize=(10, 5))
fpr, tpr, _ = roc_curve(ytest, lr_prob)
plt.plot(fpr, tpr, label='Logistic Regression', color='green')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


print(" GRADIENT BOOSTING ")

gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(xtrain_scaled, ytrain)
gb_pred = gb_model.predict(xtest_scaled)
gb_prob = gb_model.predict_proba(xtest_scaled)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(ytest, gb_pred))
print(classification_report(ytest, gb_pred))


plt.figure(figsize=(10, 5))
fpr, tpr, _ = roc_curve(ytest, gb_prob)
plt.plot(fpr, tpr, label='Gradient Boosting', color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

import pickle
with open("project/customer_churn_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)