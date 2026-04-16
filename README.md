# 🧠 ChurnSense — Agentic AI Customer Retention Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customerchurnprediction-capstone.streamlit.app/)

ChurnSense is a premium, interactive web application that combines **Machine Learning churn prediction** with an **Agentic AI Retention Strategy Assistant**. It autonomously reasons about customer risk, retrieves best practices via RAG, and generates structured intervention reports — all powered by **LangGraph**, **FAISS**, and **Google Generative AI (Gemini)**.

---

## ✨ Features

### Milestone 1 — Churn Prediction
- **Dual Model Support**: Choose between Logistic Regression and Decision Tree classifiers for instant churn probability.
- **Dynamic Real-Time Updates**: Predictions update instantly as you adjust customer metrics in the sidebar.
- **Key Risk Drivers**: Automatically identifies and explains the primary factors influencing churn (tenure, cost, support friction).
- **Model Evaluation Dashboard**: Confusion matrices, feature importance charts, and comparative model metrics.

### Milestone 2 — Agentic AI Retention Strategy Assistant
- **LangGraph Workflow**: Explicit 3-node state machine (`Analyze Risk → Retrieve Strategies → Generate Report`) with typed state management across all steps.
- **RAG with FAISS**: Embeds a curated knowledge base of retention best practices into a FAISS vector store using Google Generative AI embeddings; semantically retrieves the top-3 most relevant strategies for each customer profile.
- **Structured Output**: Every report follows a strict format:
  - 🔍 **Risk Summary** — Customer churn profile analysis
  - 💡 **Recommendations** — 3 actionable retention interventions
  - 📚 **Sources** — Best practices & references that informed the recommendations
  - ⚖️ **Disclaimer** — Business & ethical disclosures
- **Anti-Hallucination Prompting**: Low-temperature generation with explicit grounding instructions — the LLM is constrained to only reference retrieved strategies and provided customer data.
- **Premium UI**: Glassmorphic dark theme, gradient typography, animated pipeline visualization, and responsive layout.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **Gemini API Key** — Get one free from [Google AI Studio](https://aistudio.google.com/apikey) (required for Agentic AI tab)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/harshitazzz/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction

# 2. Install dependencies
pip install -r requirements.txt
```

**Dependencies include:**
`streamlit`, `pandas`, `scikit-learn`, `plotly`, `joblib`, `langchain`, `langgraph`, `langchain-google-genai`, `langchain-community`, `faiss-cpu`, `python-dotenv`

### Training the Models

Before running the app, generate the ML artifacts:

```bash
python src/train_model.py
```

This processes `data/Customer-Churn.csv`, trains both models, and saves artifacts to `models/`.

### Running the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## 🤖 Using the Agentic AI

1. Adjust customer metrics in the **sidebar** (tenure, charges, support calls).
2. Paste your **Gemini API Key** in the sidebar under "Agentic AI".
3. Navigate to the **🤖 Agentic AI** tab.
4. Click **🚀 Generate Retention Strategy**.
5. The LangGraph pipeline will:
   - **Analyze** the customer's risk profile and drivers
   - **Retrieve** semantically similar retention strategies from the FAISS vector store
   - **Generate** a structured Markdown report using Gemini

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Streamlit UI                       │
│  ┌──────────┐  ┌──────────┐  ┌──────┐  ┌──────────┐ │
│  │ Dashboard │  │ Predict  │  │Agent │  │ Metrics  │ │
│  └──────────┘  └──────────┘  └──┬───┘  └──────────┘ │
└─────────────────────────────────┼────────────────────┘
                                  │
              ┌───────────────────▼──────────────────┐
              │         LangGraph Pipeline           │
              │  ┌─────────┐ ┌────────┐ ┌─────────┐ │
              │  │ Analyze  │→│  RAG   │→│Generate │ │
              │  │  Risk    │ │ (FAISS)│ │ Report  │ │
              │  └─────────┘ └────────┘ └─────────┘ │
              └──────────────────────────────────────┘
```

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application with 4-tab UI, custom CSS, and agent integration |
| `src/agent.py` | LangGraph state machine with 3 nodes: risk analysis, FAISS retrieval, Gemini generation |
| `src/retention_strategies.txt` | Curated knowledge base of retention best practices (RAG source) |
| `src/train_model.py` | ML pipeline: data cleaning, feature engineering, model training & evaluation |
| `data/Customer-Churn.csv` | Telco customer churn dataset (7,043 records) |
| `models/*.joblib` | Serialized models, scaler, feature columns, and evaluation metrics |
| `requirements.txt` | Project dependencies |

## 🔬 Technical Details

### Machine Learning
- **Logistic Regression**: Highly interpretable, strong generalization on test data.
- **Decision Tree** (max_depth=5): Captures non-linear patterns; feature importance extraction.
- **Feature Engineering**: Auto-calculated `AvgMonthlySpend` enhances predictive power.

### Agentic AI Stack
- **LangGraph**: Explicit `StateGraph` with typed `AgentState` for deterministic workflow control.
- **FAISS**: Ephemeral vector store built on-the-fly from the knowledge base, queried with semantic embeddings.
- **Google Generative AI**: `gemini-2.0-flash` for report generation, `embedding-001` for vector embeddings.
- **Prompting**: Temperature 0.15, explicit grounding instructions, and structured output format enforcement.

---

*Built as a premium predictive analytics & agentic AI platform for proactive customer retention.*
