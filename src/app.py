# app.py

import streamlit as st
import numpy as np
import pandas as pd
from preprocessing import load_and_preprocess_data, load_config
from models import build_gp_model
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# Page settings
st.set_page_config(page_title="Active Learning Simulator", layout="centered")

# Header
st.markdown("""
    <h1 style='text-align: center; color: #7f1d1d;'>🍷 Wine Quality - Active Learning Explorer</h1>
    <p style='text-align: center; color: gray;'>Interactively simulate active learning strategies on a real dataset</p>
""", unsafe_allow_html=True)

# --- CONFIG LOAD ---
config = load_config()

# --- SESSION STATE INIT ---
if "round" not in st.session_state:
    st.session_state.round = 0
if "labeled_idx" not in st.session_state:
    X, y = load_and_preprocess_data(split=False)
    X, y = shuffle(X, y, random_state=42)
    X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    st.session_state.X_pool = X_pool
    st.session_state.y_pool = y_pool.to_numpy()
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test.to_numpy()

    seed_size = config["active_learning"]["initial_label_size"]
    all_idx = np.arange(len(X_pool))
    st.session_state.labeled_idx = np.random.choice(all_idx, size=seed_size, replace=False).tolist()
    st.session_state.unlabeled_idx = list(set(all_idx) - set(st.session_state.labeled_idx))
    st.session_state.history = []
    st.session_state.latest_uncertainty = None


# --- STRATEGY SELECTOR ---
strategy = st.radio("🎯 Choose Query Strategy", ["entropy", "least_confidence", "margin_sampling", "random"], horizontal=True)


# --- QUERY NEXT ---
if st.button("🔁 Query Next Samples"):
    model = build_gp_model()
    labeled_idx = np.array(st.session_state.labeled_idx)
    model.fit(st.session_state.X_pool[labeled_idx], st.session_state.y_pool[labeled_idx])

    # Evaluate
    y_pred = model.predict(st.session_state.X_test)
    acc = accuracy_score(st.session_state.y_test, y_pred)
    f1 = f1_score(st.session_state.y_test, y_pred, average="macro")
    st.session_state.history.append((len(labeled_idx), acc, f1))

    # Query new samples
    unlabeled_idx = np.array(st.session_state.unlabeled_idx)
    query_size = config["active_learning"]["query_batch_size"]

    proba = model.predict_proba(st.session_state.X_pool[unlabeled_idx])
    uncertainty = np.zeros(len(unlabeled_idx))  # default

    if strategy == "entropy":
        uncertainty = entropy(proba.T)
        query_indices = np.argsort(uncertainty)[-query_size:]

    elif strategy == "least_confidence":
        confidence = np.max(proba, axis=1)
        uncertainty = 1 - confidence
        query_indices = np.argsort(uncertainty)[-query_size:]

    elif strategy == "margin_sampling":
        sorted_probs = np.sort(proba, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        uncertainty = -margin  # smaller margin = higher uncertainty
        query_indices = np.argsort(uncertainty)[-query_size:]

    elif strategy == "random":
        query_indices = np.random.choice(len(unlabeled_idx), size=query_size, replace=False)
        uncertainty = np.zeros(len(unlabeled_idx))  # dummy

    else:
        st.error("Invalid strategy selected")
        st.stop()

    selected = unlabeled_idx[query_indices]
    st.session_state.labeled_idx.extend(selected.tolist())
    st.session_state.unlabeled_idx = list(set(unlabeled_idx) - set(selected))
    st.session_state.latest_uncertainty = uncertainty[query_indices]
    st.session_state.round += 1


# --- METRICS ---
col1, col2, col3 = st.columns(3)
col1.metric("🔁 Round", st.session_state.round)
col2.metric("🧪 Labeled", len(st.session_state.labeled_idx))
col3.metric("📭 Unlabeled", len(st.session_state.unlabeled_idx))


# --- UNCERTAINTY INSPECTION ---
if st.session_state.latest_uncertainty is not None:
    with st.expander("🔍 Latest Queried Sample Uncertainty"):
        st.write(pd.DataFrame({
            "Queried Index": np.arange(len(st.session_state.latest_uncertainty)),
            "Uncertainty Score": st.session_state.latest_uncertainty
        }).sort_values("Uncertainty Score", ascending=False))


# --- PLOT LEARNING CURVE ---
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history, columns=["Labeled", "Accuracy", "F1"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Labeled"], y=df["Accuracy"], mode="lines+markers", name="Accuracy"))
    fig.add_trace(go.Scatter(x=df["Labeled"], y=df["F1"], mode="lines+markers", name="F1 Score"))
    fig.update_layout(title="📈 Learning Curve", xaxis_title="Labeled Samples", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)


# --- RESET ---
st.divider()
if st.button("🔄 Reset"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()
