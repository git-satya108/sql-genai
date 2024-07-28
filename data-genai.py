import os
import openai
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from sqlalchemy import create_engine
from werkzeug.utils import secure_filename

# Load OpenAI API key
load_dotenv(find_dotenv(), override=True)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Initialize OpenAI API client
client = openai.Client()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display the banner image
st.image("imagebanner1.png", use_column_width=True)


# Function to load and read multiple Excel files
def load_excel_files(uploaded_files):
    all_sheets = {}
    for uploaded_file in uploaded_files:
        xls = pd.ExcelFile(uploaded_file)
        for sheet_name in xls.sheet_names:
            all_sheets[sheet_name] = pd.read_excel(xls, sheet_name)
    return all_sheets


# Function to break content into chunks
def break_into_chunks(text, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)
    return chunks


# Initialize FAISS index with embeddings
def initialize_faiss_index(chunks):
    embeddings = OpenAIEmbeddings()
    texts = [chunk for chunk in chunks]
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore


# Chat with the assistant using OpenAI API
def chat_with_assistant(prompt, system_message):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return {"success": False, "error": str(e)}


# Data analysis and summarization function
def analyze_data(sheets):
    try:
        response = ""
        for sheet_name, df in sheets.items():
            # Number of rows and columns
            response += f"Table '{sheet_name}' has {df.shape[0]} rows and {df.shape[1]} columns.\n\n"

            # Check if data is structured
            if df.isnull().values.any():
                response += f"Warning: The sheet '{sheet_name}' contains missing values. This might affect SQL generation.\n"

            # Natural language explanation
            prompt = f"Explain the contents of the following table:\n{df.head()}"
            explanation = chat_with_assistant(prompt,
                                              "You are a helpful assistant, SQL programmer, data scientist, and generative AI specialist.")
            response += f"Explanation: {explanation}\n\n"

        # Recommendations for unstructured data
        for sheet_name, df in sheets.items():
            if df.isnull().values.any():
                response += f"Recommendation: Please ensure the sheet '{sheet_name}' is structured without missing values for optimal SQL generation.\n"

        return response

    except Exception as e:
        return str(e)


# Streamlit app layout
st.title("SQL Query Generator")

# Sidebar for chat history
st.sidebar.title("Chat History")
for entry in st.session_state.chat_history:
    st.sidebar.write(f"User: {entry['user']}")
    st.sidebar.write(f"SQL Generator: {entry['generator']}")
    st.sidebar.write("*" * 30)

# Multiple Excel file upload
uploaded_files = st.file_uploader("Upload your Excel files", type=["xlsx"], accept_multiple_files=True)
if uploaded_files:
    sheets = load_excel_files(uploaded_files)
    if sheets:
        document_text = "\n".join([sheet.to_string() for sheet in sheets.values()])
        chunks = break_into_chunks(document_text)
        vectorstore = initialize_faiss_index(chunks)
        st.session_state['vectorstore'] = vectorstore
        st.success("Documents uploaded and processed successfully.")

# Add Data button
if st.button("Add Data"):
    vectorstore = st.session_state.get('vectorstore', None)
    if vectorstore:
        st.success("Data added to the vector store.")
    else:
        st.error("No data to add to the vector store. Please upload a document first.")

# Analyze data button
if st.button("Analyze Data"):
    if uploaded_files:
        analysis_result = analyze_data(sheets)
        st.write(analysis_result)
        for sheet_name, df in sheets.items():
            st.write(f"Preview of sheet: {sheet_name}")
            st.dataframe(df.head())
    else:
        st.error("No data to analyze. Please upload a document first.")

# SQL query generation section
st.markdown("## Generate SQL queries based on the uploaded data or provided schema:")
prompt = st.text_area("Enter your prompt here:", height=100)
if st.button("Generate SQL Query"):
    if prompt:
        sql_prompt = f"Generate an SQL query for the following request:\n{prompt}"
        sql_result = chat_with_assistant(sql_prompt,
                                         "You are a helpful assistant, SQL programmer, data scientist, and generative AI specialist.")
        if sql_result:
            st.write(sql_result)
            st.session_state.chat_history.append({
                "user": prompt,
                "generator": sql_result
            })
        else:
            st.error("No SQL query generated. Please try again.")

# Adjust prompt box and buttons
st.markdown("""
    <style>
        .stTextArea textarea {
            width: 700px;
        }
        .stButton button {
            width: 200px;
        }
    </style>
""", unsafe_allow_html=True)
