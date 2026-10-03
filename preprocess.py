"""Shared preprocessing so training and the app use the SAME logic."""
import pandas as pd

CAT_COLS = ["Employment_Status", "Employer_Category", "Loan_Purpose", "Property_Area"]
NUM_COLS = [
    "Applicant_Income", "Coapplicant_Income", "Age", "Dependents", "Existing_Loans",
    "DTI_Ratio", "Savings", "Collateral_Value", "Loan_Amount", "Loan_Term",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-value handling, copied from your notebook (training only)."""
    df = df.copy()
    if "Applicant_ID" in df.columns:
        df = df.drop(columns="Applicant_ID")

    df["Marital_Status"] = df["Marital_Status"].fillna(df["Marital_Status"].mode()[0])

    emp_na = df["Employment_Status"].isna()
    df.loc[emp_na & df["Employer_Category"].isin(["Private", "Government", "MNC"]), "Employment_Status"] = "Salaried"
    df.loc[emp_na & (df["Employer_Category"] == "Unemployed"), "Employment_Status"] = "Unemployed"
    df.loc[emp_na & (df["Employer_Category"] == "Business"), "Employment_Status"] = "Self-employed"
    df = df.drop(df[df["Employment_Status"].isna() & df["Employer_Category"].isna()].index)

    cat_na = df["Employer_Category"].isna()
    df.loc[cat_na & df["Employment_Status"].isin(["Salaried", "Contract"]), "Employer_Category"] = "Private"
    df.loc[cat_na & (df["Employment_Status"] == "Self-employed"), "Employer_Category"] = "Business"
    df.loc[cat_na & (df["Employment_Status"] == "Unemployed"), "Employer_Category"] = "Unemployed"

    df["Credit_Score_Missing"] = df["Credit_Score"].isna().astype(int)
    df["Credit_Score"] = df["Credit_Score"].fillna(df["Credit_Score"].median())

    df = df.dropna(subset=["Loan_Approved"])

    for col in NUM_COLS:
        df[col] = df[col].fillna(df[col].median())
    for col in ["Loan_Purpose", "Property_Area", "Education_Level", "Gender"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Encoding from your notebook (training only)."""
    df = df.copy()
    df["Marital_Status"] = df["Marital_Status"].map({"Single": 0, "Married": 1})
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df["Education_Level"] = df["Education_Level"].map({"Not Graduate": 0, "Graduate": 1})
    df["Loan_Approved"] = df["Loan_Approved"].map({"Yes": 1, "No": 0})
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True, dtype=int)
    df["total"] = df["Collateral_Value"] + df["Loan_Amount"]
    return df


def encode_single(inputs: dict, feature_cols: list) -> pd.DataFrame:
    """Turn ONE applicant (dict from the UI) into the exact column layout the model was trained on."""
    row = dict(inputs)
    row["Marital_Status"] = {"Single": 0, "Married": 1}[row["Marital_Status"]]
    row["Gender"] = {"Male": 0, "Female": 1}[row["Gender"]]
    row["Education_Level"] = {"Not Graduate": 0, "Graduate": 1}[row["Education_Level"]]
    row["total"] = row["Collateral_Value"] + row["Loan_Amount"]
    # manual one-hot (get_dummies on 1 row with drop_first would drop everything)
    for c in CAT_COLS:
        row[f"{c}_{row.pop(c)}"] = 1
    return pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)
