import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os


save_dir = r"D:\Code Base\Code Base\Streamlit\eday"


df = pd.read_csv(os.path.join(save_dir, 'preprocessed_diabetes.csv'))
heart_data = pd.read_csv(os.path.join(save_dir, 'preprocessed_heart.csv'))
df3 = pd.read_csv(os.path.join(save_dir, 'preprocessed_kidney.csv'))


X_diabetes = df.drop('Outcome', axis=1)
y_diabetes = df['Outcome']
X_heart = heart_data.drop('num', axis=1)
y_heart = heart_data['num']
X_kidney = df3.drop('Target', axis=1)
y_kidney = df3['Target']


scaler_diabetes = joblib.load(os.path.join(save_dir, 'scaler_diabetes.pkl'))
scaler_heart = joblib.load(os.path.join(save_dir, 'scaler_heart.pkl'))
scaler_kidney = joblib.load(os.path.join(save_dir, 'scaler_kidney.pkl'))

X_diabetes_scaled = scaler_diabetes.transform(X_diabetes)
X_heart_scaled = scaler_heart.transform(X_heart)
X_kidney_scaled = scaler_kidney.transform(X_kidney)


X_diabetes_train, X_diabetes_test, y_diabetes_train, y_diabetes_test = train_test_split(
    X_diabetes_scaled, y_diabetes, test_size=0.2, random_state=42
)
X_heart_train, X_heart_test, y_heart_train, y_heart_test = train_test_split(
    X_heart_scaled, y_heart, test_size=0.2, random_state=42
)
X_kidney_train, X_kidney_test, y_kidney_train, y_kidney_test = train_test_split(
    X_kidney_scaled, y_kidney, test_size=0.2, random_state=42
)


param_grid = {
    'n_estimators': [50, 100],    # fewer trees
    'max_depth': [None, 10],     # smaller search space
    'min_samples_split': [2, 5], 
    'min_samples_leaf': [1, 2]
}


def train_and_save_model(X_train, X_test, y_train, y_test, disease_name, cmap, filename, feature_names):
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"\nBest parameters for {disease_name}:", grid.best_params_)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    print(f"{disease_name} Optimized Accuracy:", accuracy_score(y_test, y_pred))
    print(f"{disease_name} Optimized Classification Report:\n", classification_report(y_test, y_pred))

  
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap)
    plt.title(f'{disease_name} Optimized Confusion Matrix')
    plt.savefig(os.path.join(save_dir, f'cm_{filename}_optimized.png'))
    plt.show()

    
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(f"{disease_name} Feature Importance:\n", feature_importance_df)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title(f'{disease_name} Feature Importance')
    plt.savefig(os.path.join(save_dir, f'feature_importance_{filename}.png'))
    plt.show()

    
    model_path = os.path.join(save_dir, f'model_{filename}_optimized.pkl')
    joblib.dump(best_model, model_path)
    print(f"✔ {disease_name} model saved at: {model_path}")

    return best_model


best_model_diabetes = train_and_save_model(X_diabetes_train, X_diabetes_test, y_diabetes_train, y_diabetes_test,
                                          "Diabetes", "Blues", "diabetes", X_diabetes.columns.tolist())

best_model_heart = train_and_save_model(X_heart_train, X_heart_test, y_heart_train, y_heart_test,
                                        "Heart", "Reds", "heart", X_heart.columns.tolist())

best_model_kidney = train_and_save_model(X_kidney_train, X_kidney_test, y_kidney_train, y_kidney_test,
                                         "Kidney", "Greens", "kidney", X_kidney.columns.tolist())