import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# viewing data
print(df.head())
print(df.info())
print(df.isnull().sum())

# Drop 'customerID' column
df.drop("customerID", axis=1, inplace=True)

# Convert 'TotalCharges' to numbers, then drop missing rows
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Encode text columns
label_cols = df.select_dtypes(include="object").columns
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Scale numeric columns
scaler = StandardScaler()
df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.fit_transform(df[["tenure", "MonthlyCharges", "TotalCharges"]])

# Save cleaned dataset
df.to_csv("preprocessed_data.csv", index=False)

print("Preprocessing complete.")
