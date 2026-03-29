# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and inspect its structure.

2.Separate input features and target variable, then scale the features.

3.Encode categorical target labels using LabelEncoder.

4.Split the dataset into training and testing sets.

5.Train Logistic Regression model and evaluate using accuracy, classification report, and confusion matrix.
## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: ROSHAN V
RegisterNumber:  25004228 // 212225240124
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("food_items.csv")

print("Dataset Overview")
print(df.head())
print(df.info())

X_raw = df.iloc[:, :-1]
y_raw = df.iloc[:, -1]

scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw.values.ravel())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123
)

l2_model = LogisticRegression(
    random_state=123,
    penalty='l2',
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000
)

l2_model.fit(X_train, y_train)

y_pred = l2_model.predict(X_test)

print("\nName: ROSHAN V")
print("Reg No: 212225240124")

print("\nModel Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

```

## Output:
<img width="1010" height="782" alt="image" src="https://github.com/user-attachments/assets/1ef0582c-1d80-4dc9-921a-055ca3b6a2f9" />
<img width="596" height="874" alt="Gemini_Generated_Image_bnpbwlbnpbwlbnpb" src="https://github.com/user-attachments/assets/294a39f3-992a-49a7-8ea6-0cd387bc1de2" />
<img width="1006" height="92" alt="image" src="https://github.com/user-attachments/assets/b648d4d1-0244-494e-8b6b-51d79a9f9341" />



## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
