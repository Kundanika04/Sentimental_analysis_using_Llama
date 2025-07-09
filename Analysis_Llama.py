import streamlit as st
import pandas as pd
from llama_cpp import Llama
import os
import re

# === Configuration ===
MODEL_PATH = "./Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
N_CTX = 4096
N_THREADS = 8
MAX_TOKENS_RESPONSE = 512

# --- NEW: List of potential column names to look for ---
POTENTIAL_TEXT_COLUMNS = ["Review", "Employee_Comment", "Comment", "Text", "Feedback"]

# === Load LLaMA model with caching ===
@st.cache_resource(show_spinner="Loading LLaMA model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at: {MODEL_PATH}")
        st.stop()
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
    )

llm = load_model()

# === Data Processing Function (Cached & Dynamic) ===
@st.cache_data(show_spinner="Analyzing sentiment in uploaded file...")
def process_data(uploaded_file):
    """
    Reads an Excel file, dynamically finds the text column, analyzes sentiment, 
    and returns the DataFrame and the name of the text column found.
    """
    df = pd.read_excel(uploaded_file)
    
    # --- DYNAMIC: Find the first matching text column ---
    text_column = None
    for col in POTENTIAL_TEXT_COLUMNS:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        st.error(f"File must contain one of the following columns: {', '.join(POTENTIAL_TEXT_COLUMNS)}")
        return None, None

    # --- Sentiment Analysis Step ---
    sentiments = []
    with st.status(f"Analyzing sentiment in '{text_column}' column...", expanded=True) as status:
        progress_bar = st.progress(0.0)
        total_items = len(df[text_column])

        for i, text_content in enumerate(df[text_column]):
            if pd.isna(text_content):
                sentiments.append("N/A")
                continue
            
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a sentiment analysis expert. Classify the following text as 'Positive', 'Negative', or 'Neutral'. Respond with only one of those three words.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Text: "{text_content}"
Sentiment:<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            
            output = llm(prompt, max_tokens=8, stop=["\n", "<|eot_id|>"])
            raw_sentiment = output["choices"][0]["text"].strip().capitalize()
            
            if re.search(r'\bPositive\b', raw_sentiment, re.IGNORECASE):
                sentiments.append("Positive")
            elif re.search(r'\bNegative\b', raw_sentiment, re.IGNORECASE):
                sentiments.append("Negative")
            elif re.search(r'\bNeutral\b', raw_sentiment, re.IGNORECASE):
                sentiments.append("Neutral")
            else:
                sentiments.append("Unrecognized")
            
            progress_bar.progress((i + 1) / total_items, text=f"Analyzing item {i+1}/{total_items}")
        
        status.update(label="Analysis complete!", state="complete")

    df["Sentiment"] = sentiments
    return df, text_column

# === DYNAMIC Context Retrieval Function ===
def find_relevant_context(query: str, df: pd.DataFrame, text_column: str, max_samples=5) -> str:
    """
    Finds relevant text samples from the DataFrame based on keywords in the user's query.
    This is now fully dynamic and context-agnostic.
    """
    query_lower = query.lower()
    
    # --- Step 1: Filter by sentiment if mentioned ---
    if "negative" in query_lower:
        search_df = df[df["Sentiment"] == "Negative"]
    elif "positive" in query_lower:
        search_df = df[df["Sentiment"] == "Positive"]
    else:
        search_df = df

    # --- Step 2: Extract keywords from the query to search for ---
    # A simple stopword list to make keyword search more relevant
    stop_words = set(["i", "me", "my", "is", "a", "an", "the", "and", "what", "are", "about", "show", "tell", "of", "in", "on"])
    query_keywords = [word for word in re.findall(r'\b\w+\b', query_lower) if word not in stop_words and len(word) > 2]
    
    if query_keywords:
        # Search for any of the keywords in the text column
        keyword_mask = search_df[text_column].str.contains('|'.join(query_keywords), case=False, na=False)
        relevant_df = search_df[keyword_mask]
    else:
        # If no keywords, just use the sentiment-filtered dataframe
        relevant_df = search_df

    # If filtering results in an empty DataFrame, fall back to a general sample
    if relevant_df.empty:
        relevant_df = search_df if not search_df.empty else df
        
    num_samples = min(max_samples, len(relevant_df))
    if num_samples == 0:
        return "No relevant data found for this query."
        
    sample_df = relevant_df.sample(n=num_samples, random_state=42)

    context_str = "Here are some relevant data samples:\n\n"
    for _, row in sample_df.iterrows():
        context_str += f"- Sentiment: {row['Sentiment']}\n  {text_column}: \"{row[text_column]}\"\n\n"
        
    return context_str.strip()

# === DYNAMIC System Prompt Generation ===
def get_system_prompt(text_column: str, data_summary: str, relevant_context: str) -> str:
    """Generates a system prompt with a persona adapted to the data context."""
    persona = "a helpful Data Analyst Assistant"
    if text_column == "Employee_Comment":
        persona = "an insightful HR Analyst Assistant"
    elif text_column == "Review":
        persona = "a helpful Customer Feedback Analyst"

    return f"""You are {persona}. Your task is to answer user questions based on the provided data.
- Base your answers strictly on the 'Relevant Data Samples' and the 'Data Summary'.
- Synthesize information from multiple samples to identify key themes.
- If the data doesn't contain an answer, state that clearly. Do not invent information.

---
DATA SUMMARY:
{data_summary}
---
RELEVANT DATA SAMPLES:
{relevant_context}
---
"""

# === Streamlit App Layout ===
st.set_page_config(page_title="Contextual Data Chat", layout="wide")
st.title("📄💬 Chat with your Data")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_data" not in st.session_state:
    st.session_state.processed_data = {"df": None, "text_column": None}

# === Sidebar for File Upload and Data Display ===
with st.sidebar:
    st.header("Upload & Analyze Data")
    uploaded_file = st.file_uploader(
        f"Upload an Excel file with a text column (e.g., {', '.join(POTENTIAL_TEXT_COLUMNS)})",
        type=["xlsx"],
        key="file_uploader"
    )

    if uploaded_file:
        df, text_col = process_data(uploaded_file)
        if df is not None and text_col is not None:
            st.session_state.processed_data["df"] = df
            st.session_state.processed_data["text_column"] = text_col
            st.success(f"File processed! Using '{text_col}' column.")

    processed_df = st.session_state.processed_data["df"]
    if processed_df is not None:
        st.subheader("Sentiment Analysis Summary")
        summary_counts = processed_df["Sentiment"].value_counts()
        st.bar_chart(summary_counts)
        st.subheader("Analyzed Data")
        st.dataframe(processed_df)

# === Main Chat Interface ===
st.subheader("Ask Anything About Your Data")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question about your data..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    df = st.session_state.processed_data["df"]
    text_column = st.session_state.processed_data["text_column"]

    if df is None:
        reply = "I'm ready to help, but you need to upload an Excel file first."
    else:
        with st.spinner("Thinking..."):
            relevant_context = find_relevant_context(user_input, df, text_column)
            data_summary = df['Sentiment'].value_counts().to_string()
            
            system_prompt = get_system_prompt(text_column, data_summary, relevant_context)
            
            messages_for_llm = [
                {"role": "system", "content": system_prompt}
            ] + st.session_state.chat_history

            output = llm.create_chat_completion(
                messages=messages_for_llm,
                max_tokens=MAX_TOKENS_RESPONSE,
                stop=["<|eot_id|>"],
                temperature=0.7,
            )
            reply = output['choices'][0]['message']['content'].strip()

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
