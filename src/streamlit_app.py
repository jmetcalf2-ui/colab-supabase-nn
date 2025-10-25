import os
import pandas as pd
import streamlit as st
from supabase import create_client
import joblib
import openai

# Load environment variables for Supabase and OpenAI
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

@st.cache_data(show_spinner=False)
def init_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("Supabase credentials are not set. Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables.")
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_data(show_spinner=False)
def load_leads(client):
    if client is None:
        return pd.DataFrame()
    data = client.table("leads").select("*").execute().data
    df = pd.DataFrame(data)
    return df

@st.cache_resource(show_spinner=False)
def load_model():
    # Load baseline model pipeline (preprocessor + classifier) saved in models folder
    try:
        model = joblib.load("models/baseline_model.joblib")
        return model
    except Exception:
        return None

def render_dashboard():
    st.header("Leads Dashboard")
    client = init_supabase()
    df = load_leads(client)
    if df.empty:
        st.write("No data available or failed to load.")
        return
    st.write(f"Total leads: {len(df)}")
    st.dataframe(df)
    if "primary_role" in df.columns:
        st.subheader("Primary Role Distribution")
        st.bar_chart(df["primary_role"].value_counts())

def render_ai_tab():
    st.header("Predict Primary Role")
    model = load_model()
    if model is None:
        st.write("Model not found. Please ensure baseline_model.joblib is in the models directory.")
        return
    client = init_supabase()
    df = load_leads(client)
    if df.empty:
        st.write("Cannot determine feature columns without data.")
        return
    feature_cols = [c for c in df.columns if c not in ["primary_role", "lead_id", "full_name"]]
    user_input = {}
    for col in feature_cols:
        user_input[col] = st.text_input(col, "")
    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted primary role: {prediction}")

def render_chat_tab():
    st.header("AI Chat")
    if not OPENAI_API_KEY:
        st.write("OpenAI API key is not set. Please set OPENAI_API_KEY environment variable.")
        return
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display messages
    for msg in st.session_state.messages:
        role = "You" if msg["role"] == "user" else "AI"
        st.markdown(f"**{role}:** {msg['content']}")
    user_text = st.text_input("Your message:", "")
    if st.button("Send"):
        if user_text:
            st.session_state.messages.append({"role": "user", "content": user_text})
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state.messages
                )
                reply = response.choices[0].message["content"].strip()
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.write("Error contacting OpenAI API: " + str(e))

def main():
    st.title("Supabase Leads Dashboard & AI Prediction/Chat")
    tabs = st.tabs(["Leads Dashboard", "AI Prediction", "AI Chat"])
    with tabs[0]:
        render_dashboard()
    with tabs[1]:
        render_ai_tab()
    with tabs[2]:
        render_chat_tab()

if __name__ == "__main__":
    main()
