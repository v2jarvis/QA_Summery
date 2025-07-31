import streamlit as st
import docx2txt
import fitz  # PyMuPDF
import re
import os
import requests

st.set_page_config(page_title="Health Questionnaire Evaluator", layout="wide")
st.title("🩺 Health Questionnaire Evaluator")

st.markdown("""
Upload a **health questionnaire** in PDF or DOCX format where each question has a **Yes/No answer**.  
The system will analyze the answers and return structured insights using **AI**.

**Output Format Includes:**
- Summary  
- Suggestions  
- Follow-up  
- Diagnosis
""")

uploaded_file = st.file_uploader("📄 Upload your Health Questionnaire", type=["pdf", "docx"])

# --- File Handlers ---
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_docx(file):
    return docx2txt.process(file)

# --- QA Parser ---
def parse_yes_no_questions(text):
    qa_pairs = re.findall(r'•\s+(.*?)\s*Answer:\s*(Yes|No)', text, re.DOTALL)
    return "\n".join([f"Q: {q.strip()} | A: {a}" for q, a in qa_pairs]), qa_pairs

# --- Prompt Constructor ---
def build_chain_of_thought_prompt(qa_text):
    return f"""
🧑‍⚕️ **Role**:
You are a compassionate medical assistant helping interpret Yes/No health questionnaires in natural, empathetic language.

📜 **Rules**:
- Do not use bullet points.
- Do not summarize in a list format.
- Avoid headings like "Clinical Hypothesis" or enumerated criteria.
- Keep tone human, supportive, and emotionally intelligent.
- Structure the response in **plain paragraph-style prose**, just like a doctor would speak during a thoughtful consultation.

🎯 **Objective**:
From the following questionnaire responses, write a structured health response that includes:

**Summary** – A narrative of what the person is going through, based on their answers.

**Suggestions** – Empathetic advice and treatment possibilities, written gently and supportively.

**Follow-up** – Invite the person to take next steps, like speaking with a provider or attending a follow-up.

**Diagnosis** – Hypothesize possible health conditions based on answer patterns, explained clearly and without overwhelming the user.

---

Here are the person's questionnaire responses:

{qa_text}
"""

groq_api_key = st.secrets["groq"]["api_key"]

# --- LLM Integration ---
def call_groq_llm(prompt):
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.1-8b-instant",  
        "messages": [
            {"role": "system", "content": "You are a medical assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]

# --- Main App Logic ---
if uploaded_file:
    if uploaded_file.name.endswith("pdf"):
        file_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith("docx"):
        file_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    qa_text, pairs = parse_yes_no_questions(file_text)

    if not pairs:
        st.warning("❗ No valid Yes/No question-answer pairs found.")
    else:
        st.markdown("### 🧾 Extracted Q&A")
        st.code(qa_text)

        if st.button("🔍 Analyze with AI"):
            with st.spinner("Analyzing with AI reasoning..."):
                prompt = build_chain_of_thought_prompt(qa_text)
                try:
                    result = call_groq_llm(prompt)
                    st.success("✅ Analysis Complete")
                    st.markdown("### 🧠 Result")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"LLM call failed: {e}")
