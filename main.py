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

st.set_page_config(page_title="Resume Analyzer", layout="wide")

# Centered Title
st.markdown("<h1 style='text-align: center;'>📄 Resume Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload your <b>resume PDF</b> and paste a <b>job description</b> to get smart feedback.</p>", unsafe_allow_html=True)

# Split inputs into columns
col1, col2 = st.columns(2)

# Session state trigger for button
if "analyze_clicked" not in st.session_state:
    st.session_state["analyze_clicked"] = False

with col1:
    st.markdown("### 📎 Upload Resume (PDF)")
    resume_file = st.file_uploader(label="", type=["pdf"])

    st.markdown("### 🧾 Paste Job Description")
    job_description = st.text_area(label="", height=250)

    
# Footer-style Analyze Button (centered)
footer_col = st.container()
with footer_col:
    center_col = st.columns([1, 2, 1])[1]  # Center the button in the middle column
    with center_col:
        if st.button("🚀 Analyze Resume"):
            st.session_state["analyze_clicked"] = True

# Output in col2
with col2:
    if st.session_state["analyze_clicked"]:
        if not resume_file or not job_description.strip():
            st.warning("Please upload a resume and paste the job description.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    resume_text = extract_text_from_pdf(resume_file)
                    result = call_grok_backend(resume_text, job_description)

                    # st.success("✅ Analysis Complete")

                     # Display Match Percentage
                    match_percentage = f"{result.get('match_percentage', 0)}%"

                    #st.text_area("🎯 Match Percentage", value=match_percentage, height=80)
                   
                   # Set the bar style with thinner height
                    bar_style = f"width: {match_percentage}%; background-color: #4CAF50; height: 20px; border-radius: 10px;"
                    
                   # Display the match percentage and progress bar
                    st.markdown(f"<h4 style='font-weight:bold;'>🎯 Match Percentage</h4>", unsafe_allow_html=True)
                    st.text_area(label="", value=match_percentage, height=70)


                    # Display Missing Skills
                    if result.get("missing_skills"):
                        missing_skills_text = "\n".join(f"- {skill}" for skill in result["missing_skills"])
                    else:
                        missing_skills_text = "- No missing skills identified."
                    st.markdown(f"<h4 style='font-weight:bold;'>🔍 Missing Skills</h4>", unsafe_allow_html=True)
                    st.text_area(label="", value=missing_skills_text, height=140)


                    # Display Suggestionss
                    suggestion = result.get("suggestion", "- No suggestion provided.")
                    suggestion_bullets = "\n".join(f"- {line.strip()}" for line in suggestion.split('.') if line.strip())

                    st.markdown(f"<h4 style='font-weight:bold;'>💡 Suggestion</h4>", unsafe_allow_html=True)
                    st.text_area(label="", value=suggestion_bullets, height=120)


                # Move success message here (footer)
                    with st.container():
                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.success("✅ Analysis Complete")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
