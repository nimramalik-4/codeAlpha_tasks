import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

# Load dataset
df = sns.load_dataset("titanic")

# Data Preprocessing
df.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male'], axis=1, inplace=True)
df.dropna(inplace=True)

# Encode categorical features
label_enc = LabelEncoder()
df['sex'] = label_enc.fit_transform(df['sex'])
df['embarked'] = label_enc.fit_transform(df['embarked'])
df['alone'] = label_enc.fit_transform(df['alone'])

# Define features and target
X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'alone']]
y = df['survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Streamlit Web App
def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked, alone):
    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked, alone]])
    features = scaler.transform(features)
    prediction = model.predict(features)
    return "Survived" if prediction[0] == 1 else "Not Survived"

st.title("Titanic Survival Prediction")
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=50.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])
alone = st.selectbox("Alone", ["Yes", "No"])

sex = 1 if sex == "Male" else 0
embarked = {"C": 0, "Q": 1, "S": 2}[embarked]
alone = 1 if alone == "Yes" else 0

if st.button("Predict Survival"):
    result = predict_survival(pclass, sex, age, sibsp, parch, fare, embarked, alone)
    st.success(f"The passenger would {result}")
