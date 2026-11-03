import streamlit as st
from google import genai
from google.genai import types

# 1. Page Configuration & Header
st.set_page_config(page_title="Chat with Thomas Edison", page_icon="💡", layout="centered")
st.title("💡 Interview Thomas Edison")

# 2. Securely Get the Secret API Key from Streamlit Cloud Secrets
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception:
    st.error("Missing Gemini API Key. Please add 'GEMINI_API_KEY' to your Streamlit App Secrets.")
    st.stop()

# 3. Instruction Guide for High School Students
st.info(
    "👋 Welcome! Due to a temporary Streamlit Cloud layout bug, the normal chat box is disabled.\n\n"
    "**HOW TO ASK A QUESTION:**\n"
    "Look at the address bar at the very top of your web browser. "
    "Add your question to the very end of the URL link like this, then press Enter:\n\n"
    "`...streamlit.app/?question=your question here`"
)

# 4. Fetch the Student's Question directly from the URL link parameters
url_params = st.query_params
student_question = url_params.get("question", None)

# 5. Process and Display the Conversation
if student_question:
    st.write(f"### 👤 Your Question:")
    st.success(student_question)
    
    # Historical context system instructions
    system_instruction = (
        "You are acting as the historical figure Thomas Edison. Speak in the first person ('I'). "
        "Your tone must be high-school appropriate, educational, professional, yet determined and sharp. "
        "Answer the student's question based strictly on your historical written works, diaries, patents, "
        "and primary academic materials (like the Rutgers Edison Papers Project).\n\n"
        "CRITICAL RULES:\n"
        "1. Your primary response content MUST be between 20 and 80 words long.\n"
        "2. At the very end of your response, skip a line and provide a section titled 'Sources Cited:' "
        "listing 1 or 2 specific historical academic records (e.g., 'Edison Patent No. 223,898', "
        "'Diary entry, July 1885', or 'The Papers of Thomas A. Edison, Rutgers University')."
    )

    formatted_prompt = f"{system_instruction}\n\nStudent asks: {student_question}\nEdison:"

    with st.spinner("⚡ Mr. Edison is drafting a response from Menlo Park..."):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=formatted_prompt
            )
            st.write(f"### 🤖 Thomas Edison Replies:")
            st.write(response.text)
        except Exception as e:
            st.error(f"Connection Error: {e}")
else:
    st.warning("⚠️ No question detected yet. Go to the top URL bar and add `/?question=How did you invent the lightbulb` to start!")
