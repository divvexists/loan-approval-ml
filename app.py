import joblib
import pandas as pd
import streamlit as st

from preprocess import encode_single

st.set_page_config(page_title="CreditWise Loan System", page_icon="🏦", layout="wide")


@st.cache_resource
def load():
    bundle = joblib.load("model.joblib")
    return bundle["model"], bundle["meta"]


model, meta = load()
cats, stats = meta["categories"], meta["num_stats"]

st.title("🏦 CreditWise Loan Approval System")
st.caption("Decision-tree model trained on historical loan applications.")

tab_predict, tab_model = st.tabs(["Check an application", "Model performance"])


with tab_predict:
    with st.form("applicant"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Applicant")
            age = st.number_input("Age", 18, 100, int(stats["Age"]["median"]))
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital = st.selectbox("Marital status", ["Single", "Married"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            dependents = st.number_input("Dependents", 0, 10, int(stats["Dependents"]["median"]))
            emp_status = st.selectbox("Employment status", cats["Employment_Status"])
            emp_cat = st.selectbox("Employer category", cats["Employer_Category"])

        with c2:
            st.subheader("Finances")
            app_inc = st.number_input("Applicant income", 0.0, value=stats["Applicant_Income"]["median"], step=500.0)
            co_inc = st.number_input("Coapplicant income", 0.0, value=0.0, step=500.0)
            savings = st.number_input("Savings", 0.0, value=stats["Savings"]["median"], step=500.0)
            collateral = st.number_input("Collateral value", 0.0, value=stats["Collateral_Value"]["median"], step=1000.0)
            dti = st.number_input("DTI ratio", 0.0, value=stats["DTI_Ratio"]["median"], step=0.01)
            existing = st.number_input("Existing loans", 0, 20, int(stats["Existing_Loans"]["median"]))

        with c3:
            st.subheader("Credit & loan")
            no_score = st.checkbox("Applicant has no credit score")
            score = st.number_input("Credit score", 300, 900, int(stats["Credit_Score"]["median"]), disabled=no_score)
            loan_amt = st.number_input("Loan amount", 0.0, value=stats["Loan_Amount"]["median"], step=1000.0)
            term = st.number_input("Loan term", 1, 600, int(stats["Loan_Term"]["median"]))
            purpose = st.selectbox("Loan purpose", cats["Loan_Purpose"])
            area = st.selectbox("Property area", cats["Property_Area"])

        submitted = st.form_submit_button("Check eligibility", type="primary")

    if submitted:
        inputs = {
            "Applicant_Income": app_inc, "Coapplicant_Income": co_inc, "Employment_Status": emp_status,
            "Age": age, "Marital_Status": marital, "Dependents": dependents,
            "Credit_Score": meta["credit_score_median"] if no_score else score,
            "Existing_Loans": existing, "DTI_Ratio": dti, "Savings": savings,
            "Collateral_Value": collateral, "Loan_Amount": loan_amt, "Loan_Term": term,
            "Loan_Purpose": purpose, "Property_Area": area, "Education_Level": education,
            "Gender": gender, "Employer_Category": emp_cat,
            "Credit_Score_Missing": int(no_score),
        }
        X_row = encode_single(inputs, meta["feature_cols"])
        prob = float(model.predict_proba(X_row)[0][1])
        approved = prob >= 0.5

        st.divider()
        if approved:
            st.success(f"✅ Likely APPROVED — model confidence {prob:.0%}")
        else:
            st.error(f"❌ Likely REJECTED — approval probability {prob:.0%}")
        st.progress(prob)
        st.caption("This is a model estimate, not a final lending decision.")


with tab_model:
    m = meta["metrics"]
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{m['accuracy']:.1%}")
    b.metric("Precision", f"{m['precision']:.1%}")
    c.metric("Recall", f"{m['recall']:.1%}")
    d.metric("F1", f"{m['f1']:.1%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix (test set)")
        cm = pd.DataFrame(m["confusion_matrix"],
                          index=["Actual: No", "Actual: Yes"],
                          columns=["Pred: No", "Pred: Yes"])
        st.dataframe(cm)
        st.caption("Recall = good customers wrongly rejected. Precision = risky customers wrongly approved.")
    with right:
        st.subheader("Top features")
        imp = pd.Series(meta["importances"]).sort_values(ascending=False).head(10)
        st.bar_chart(imp)
