import streamlit as st
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
st.title("Fraud Detection Demo")

with st.form("transaction"):
    step = st.number_input("Simulation step", min_value=1, step=1)
    transaction_type = st.selectbox(
        "Transaction type",
        ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    )
    amount = st.number_input("Transaction amount", min_value=0.0, format="%.2f")
    sender_balance = st.number_input("Sender's opening balance", min_value=0.0, format="%.2f")
    recipient_balance = st.number_input("Recipient's opening balance", min_value=0.0, format="%.2f")
    submitted = st.form_submit_button("Check transaction")

if submitted:
    # Load the model and threshold saved by your notebook.
    saved = joblib.load(Path(__file__).with_name("fraud_detector.joblib"))
    model = saved["model"]

    # Build one transaction using the same calculations as training.
    transaction = pd.DataFrame([{
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": sender_balance,
        "oldbalanceDest": recipient_balance,
        "day": (step - 1) // 24 + 1,
        "hour": (step - 1) % 24,
        "balance_ratio": amount / sender_balance if sender_balance > 0 else 0,
        "empties_account": int(
            sender_balance > 0
            and np.isclose(amount, sender_balance, atol=0.01, rtol=0)
        ),
        "zero_balance": int(sender_balance == 0)
    }])

    # Score this transaction and apply your saved cutoff.
    score = model.predict_proba(transaction)[0, 1]

    st.subheader("Model decision")
    if score >= saved["threshold"]:
        st.warning("Suspicious transaction")
    else:
        st.success("Normal transaction")

    st.metric("Model fraud score", f"{score:.2%}")
    st.write("Raw score for this transaction:", repr(float(score)))
        # Follow the same decision path the model used to score this transaction.
    preprocessor, tree = model[0], model[-1]
    path = tree.decision_path(preprocessor.transform(transaction)).indices
    visited = set(path)

    money = ["amount", "oldbalanceOrg", "oldbalanceDest", "balance_ratio"]
    time = ["step", "day", "hour"]
    binary = ["empties_account", "zero_balance"]
    transformers = preprocessor.named_transformers_
    categories = transformers["category"].categories_[0]
    features = money + time + [f"type={value}" for value in categories] + binary

    labels = {
        "amount": "Transaction amount",
        "oldbalanceOrg": "Sender’s opening balance",
        "oldbalanceDest": "Recipient’s opening balance",
        "balance_ratio": "Amount as a share of sender balance",
        "step": "Simulation step",
        "day": "Simulation day",
        "hour": "Simulation hour",
        "empties_account": "Empties sender account",
        "zero_balance": "Sender balance is zero",
    }

    factors = {}
    for node in reversed(path):
        index = tree.tree_.feature[node]
        if index < 0:  # The final leaf has no split.
            continue

        column = features[index]
        cutoff = tree.tree_.threshold[node]
        went_left = tree.tree_.children_left[node] in visited

        if column.startswith("type="):
            label = "Transaction type"
            category = column.split("=", 1)[1]
            detail = f"{'differs from' if went_left else 'matches'} {category}"
        elif column in binary:
            label = labels[column]
            detail = "No" if went_left else "Yes"
        else:
            label = labels[column]
            # Convert the split back to the units shown in the form.
            if column in money:
                scaler = transformers["money"][-1]
                cutoff = np.expm1(cutoff * scaler.scale_[index] + scaler.mean_[index])
            else:
                scaler = transformers["time"]
                position = time.index(column)
                cutoff = cutoff * scaler.scale_[position] + scaler.mean_[position]

            value = transaction.at[0, column]
            fmt = ".2%" if column == "balance_ratio" else ",.2f"
            operator = "≤" if went_left else ">"
            detail = f"{value:{fmt}} (tree checked {operator} ~{cutoff:{fmt}})"

        factors.setdefault(label, detail)
        if len(factors) == 4:
            break

    st.subheader("Key factors")
    for label, detail in reversed(list(factors.items())):
        st.markdown(f"- **{label}** — {detail}")
