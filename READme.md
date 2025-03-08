# **Titanic Survival Prediction**

This project is a **Machine Learning model** that predicts whether a person will survive sinking based on various factors such as **socio-economic status, age, gender, and more**. The dataset used is the Titanic dataset.

## **🚀 Features**

* **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature scaling.  
* **Exploratory Data Analysis (EDA)**: Understanding survival trends using visualization.  
* **Feature Engineering**: Creating meaningful features.  
* **Model Training**: Using **Random Forest Classifier** for prediction.  
* **Model Evaluation**: Accuracy, classification report, confusion matrix.  
* **Web App (Streamlit)**: A user-friendly interface for prediction.

## **📌 Dataset**

The dataset is sourced from **Seaborn’s Titanic dataset** and includes the following key features:

* `pclass`: Passenger class (1st, 2nd, 3rd)  
* `sex`: Gender  
* `age`: Age of the passenger  
* `sibsp`: Number of siblings/spouses aboard  
* `parch`: Number of parents/children aboard  
* `fare`: Ticket fare paid  
* `embarked`: Port of embarkation (C, Q, S)  
* `alone`: Whether the passenger was traveling alone  
* `survived`: Target variable (1 \= Survived, 0 \= Not Survived)

## **🛠 Installation & Usage**

### **Clone the Repository**

git clone https://github.com/your-username/Titanic-Survival-Prediction.git  
cd Titanic-Survival-Prediction

### **Install Dependencies**

pip install pandas numpy seaborn matplotlib scikit-learn streamlit

### **Run the Model**

python titanic\_model.py

### **Run the Web App**

streamlit run titanic\_app.py

## **📊 Model Performance**

* **Accuracy**: \~80% (varies based on hyperparameter tuning)  
* **Confusion Matrix** and **Classification Report** included for evaluation.

## **🎯 Future Improvements**

* Try **other ML models** (XGBoost, SVM, Neural Networks).  
* Deploy using **Flask or FastAPI**.  
* Integrate a more detailed dataset.

## **🤝 Contributing**

Feel free to **fork** this repository and improve the project\!

## **📜 License**

This project is under the **MIT License**.

---

Made by Nimra

