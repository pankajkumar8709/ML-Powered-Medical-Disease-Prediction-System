import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os


os.chdir('D:\\Code Base\\Code Base\\Streamlit\\eday')


df = pd.read_csv('preprocessed_diabetes.csv')
heart_data = pd.read_csv('preprocessed_heart.csv')
df3 = pd.read_csv('preprocessed_kidney.csv')


X_diabetes = df.drop('Outcome', axis=1)
y_diabetes = df['Outcome']
X_heart = heart_data.drop('num', axis=1)
y_heart = heart_data['num']
X_kidney = df3.drop('Target', axis=1)
y_kidney = df3['Target']


scaler_diabetes = joblib.load('D:\\Code Base\\Code Base\\Streamlit\\eday\\scaler_diabetes.pkl')
scaler_heart = joblib.load('D:\\Code Base\\Code Base\\Streamlit\\eday\\scaler_heart.pkl')
scaler_kidney = joblib.load('D:\\Code Base\\Code Base\\Streamlit\\eday\\scaler_kidney.pkl')


X_diabetes_scaled = scaler_diabetes.transform(X_diabetes)
X_heart_scaled = scaler_heart.transform(X_heart)
X_kidney_scaled = scaler_kidney.transform(X_kidney)


from sklearn.model_selection import train_test_split
X_diabetes_train, X_diabetes_test, y_diabetes_train, y_diabetes_test = train_test_split(X_diabetes_scaled, y_diabetes, test_size=0.2, random_state=42)
X_heart_train, X_heart_test, y_heart_train, y_heart_test = train_test_split(X_heart_scaled, y_heart, test_size=0.2, random_state=42)
X_kidney_train, X_kidney_test, y_kidney_train, y_kidney_test = train_test_split(X_kidney_scaled, y_kidney, test_size=0.2, random_state=42)


model_diabetes = RandomForestClassifier(n_estimators=100, random_state=42)
model_diabetes.fit(X_diabetes_train, y_diabetes_train)
y_diabetes_pred = model_diabetes.predict(X_diabetes_test)
print("Diabetes Accuracy:", accuracy_score(y_diabetes_test, y_diabetes_pred))
print("Diabetes Classification Report:\n", classification_report(y_diabetes_test, y_diabetes_pred))
cm_diabetes = confusion_matrix(y_diabetes_test, y_diabetes_pred)
sns.heatmap(cm_diabetes, annot=True, fmt='d', cmap='Blues')
plt.title('Diabetes Confusion Matrix')
plt.savefig('cm_diabetes.png')
plt.show()


model_heart = RandomForestClassifier(n_estimators=100, random_state=42)
model_heart.fit(X_heart_train, y_heart_train)
y_heart_pred = model_heart.predict(X_heart_test)
print("Heart Accuracy:", accuracy_score(y_heart_test, y_heart_pred))
print("Heart Classification Report:\n", classification_report(y_heart_test, y_heart_pred))
cm_heart = confusion_matrix(y_heart_test, y_heart_pred)
sns.heatmap(cm_heart, annot=True, fmt='d', cmap='Reds')
plt.title('Heart Confusion Matrix')
plt.savefig('cm_heart.png')
plt.show()

model_kidney = RandomForestClassifier(n_estimators=100, random_state=42)
model_kidney.fit(X_kidney_train, y_kidney_train)
y_kidney_pred = model_kidney.predict(X_kidney_test)
print("Kidney Accuracy:", accuracy_score(y_kidney_test, y_kidney_pred))
print("Kidney Classification Report:\n", classification_report(y_kidney_test, y_kidney_pred))
cm_kidney = confusion_matrix(y_kidney_test, y_kidney_pred)
sns.heatmap(cm_kidney, annot=True, fmt='d', cmap='Greens')
plt.title('Kidney Confusion Matrix')
plt.savefig('cm_kidney.png')
plt.show()


joblib.dump(model_diabetes, 'D:\\Code Base\\Code Base\\Streamlit\\eday\\model_diabetes.pkl')
joblib.dump(model_heart, 'D:\\Code Base\\Code Base\\Streamlit\\eday\\model_heart.pkl')
joblib.dump(model_kidney, 'D:\\Code Base\\Code Base\\Streamlit\\eday\\model_kidney.pkl')