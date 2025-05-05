# main.py

import streamlit as st
import requests
import json
import os
from pypdf import PdfReader
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# Extracts text from uploaded PDF
def extract_text_from_pdf(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()]).strip()

# Calls the Groq API for resume evaluation
def call_grok_backend(resume_text: str, jd_text: str) -> dict:
    if not API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    prompt = f"""
Resume:
{resume_text}

Job Description:
{jd_text}
"""

    system_prompt = """
You are a resume screening expert. Given a resume and job description, perform the following:

1. Estimate a match percentage based on the alignment of the resume to the job description.
2. Identify key technical or soft skills missing from the resume.
3. Give a 2-line suggestion to improve the resume.

⚠️ Format your output strictly as a JSON object like this:
{
  "match_percentage": <number>,
  "missing_skills": ["skill1", "skill2"],
  "suggestion": "<your suggestion here>"
}
""".strip()

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",  # ✅ Replace with model Groq supports
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")

    try:
        content = response.json()['choices'][0]['message']['content']
        return json.loads(content)  # Must return valid JSON
    except (KeyError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to parse model response as JSON:\n{response.text}")

# ───────────────────── Streamlit UI ─────────────────────

st.set_page_config(page_title="Resume Analyzer", layout="centered")
st.title("📄 Resume Analyzer")
st.markdown("Upload your **resume PDF** and paste a **job description** to get smart feedback.")

resume_file = st.file_uploader("📎 Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("🧾 Paste Job Description", height=250)

if st.button("🚀 Analyze Resume"):
    if not resume_file or not job_description.strip():
        st.warning("Please upload a resume and paste the job description.")
    else:
        with st.spinner("Analyzing..."):
            try:
                resume_text = extract_text_from_pdf(resume_file)
                result = call_grok_backend(resume_text, job_description)

                st.success("✅ Analysis Complete")
                st.metric("🎯 Match Percentage", f"{result.get('match_percentage', 'N/A')}%")
                st.write("🔍 **Missing Skills:**")
                st.write(", ".join(result.get("missing_skills", [])))
                st.write("💡 **Suggestion:**")
                st.info(result.get("suggestion", "No suggestion provided."))

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
