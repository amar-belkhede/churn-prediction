import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import pickle

telecom_cust = pd.read_csv('Telco-Customer-Churn.csv')

# Converting Total Charges to a numerical data type.
telecom_cust.TotalCharges = pd.to_numeric(
    telecom_cust.TotalCharges, errors='coerce')
telecom_cust.isnull().sum()

# Removing missing values
telecom_cust.dropna(inplace=True)
# Remove customer IDs from the data set
df2 = telecom_cust.iloc[:, 1:]
# Convertin the predictor variable in a binary numeric variable
df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df2['Churn'].replace(to_replace='No',  value=0, inplace=True)

# Let's convert all the categorical variables into dummy variables
df_dummies = pd.get_dummies(df2)
df_dummies.head()

# We will use the data frame where we had created dummy variables
y = df_dummies['Churn'].values
X = df_dummies.drop(columns=['Churn'])

# # Scaling all the variables to a range of 0 to 1
# from sklearn.preprocessing import MinMaxScaler
# features = X.columns.values
# scaler = MinMaxScaler(feature_range = (0,1))
# scaler.fit(X)
# X = pd.DataFrame(scaler.transform(X))
# X.columns = features

# Create Train & Test Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

# from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)
xgb = metrics.accuracy_score(y_test, preds)
print(metrics.accuracy_score(y_test, preds))


# Creating a dump file of the model
pickle.dump(model, open('model.pkl', 'wb'))

# Loading the model
model = pickle.load(open('model.pkl', 'rb'))

# Test
resJsonNorm = {
    "SeniorCitizen": 0,
    "tenure": 34,
    "MonthlyCharges": 56.95,
    "TotalCharges": 1889.50,
    "gender_Female": 0,
    "gender_Male": 1,
    "Partner_No": 1,
    "Partner_Yes": 0,
    "Dependents_No": 1,
    "Dependents_Yes": 0,
    "PhoneService_No": 0,
    "PhoneService_Yes": 1,
    "MultipleLines_No": 1,
    "MultipleLines_No phone service": 0,
    "MultipleLines_Yes": 0,
    "InternetService_DSL": 1,
    "InternetService_Fiber optic": 0,
    "InternetService_No": 0,
    "OnlineSecurity_No": 0,
    "OnlineSecurity_No internet service": 0,
    "OnlineSecurity_Yes": 1,
    "OnlineBackup_No": 1,
    "OnlineBackup_No internet service": 0,
    "OnlineBackup_Yes": 0,
    "DeviceProtection_No": 0,
    "DeviceProtection_No internet service": 0,
    "DeviceProtection_Yes": 1,
    "TechSupport_No": 1,
    "TechSupport_No internet service": 0,
    "TechSupport_Yes": 0,
    "StreamingTV_No": 1,
    "StreamingTV_No internet service": 0,
    "StreamingTV_Yes": 0,
    "StreamingMovies_No": 1,
    "StreamingMovies_No internet service": 0,
    "StreamingMovies_Yes": 0,
    "Contract_Month-to-month": 0,
    "Contract_One year": 1,
    "Contract_Two year": 0,
    "PaperlessBilling_No": 1,
    "PaperlessBilling_Yes": 0,
    "PaymentMethod_Bank transfer (automatic)": 0,
    "PaymentMethod_Credit card (automatic)": 0,
    "PaymentMethod_Electronic check": 0,
    "PaymentMethod_Mailed check": 1
}

resNorm = pd.DataFrame(resJsonNorm, index=[0])
print(model.predict(resNorm))
