import streamlit as st
from google import genai
from google.genai import types

# 1. Page Configuration & Title
st.set_page_config(page_title="Chat with Thomas Edison", page_icon="💡", layout="centered")
st.title("💡 Interview Thomas Edison")
st.write("Ask Mr. Edison a question about his inventions, life, or process. Responses are backed by historical materials.")

# 2. Securely Get the Secret API Key from Streamlit Cloud
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception:
    st.error("Missing Gemini API Key. Please add 'GEMINI_API_KEY' to your Streamlit App Secrets.")
    st.stop()

# 3. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Greetings. I am Thomas Alva Edison. Ask me anything about my work at Menlo Park, my patents, or my view on innovation."}
    ]

# 4. Display Past Messages Using Standard Text Containers (Bypasses Glitchy Chat Bubbles)
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.markdown(f"**🤖 Thomas Edison:** {msg['content']}")
    else:
        st.markdown(f"**👤 Student:** {msg['content']}")
st.write("---")

# 5. Fallback Input Form (Bypasses Broken st.chat_input script completely)
with st.form(key="edison_input_form", clear_on_submit=True):
    user_query = st.text_input("Type your question for Mr. Edison here and press 'Submit':", placeholder="e.g., How did you invent the lightbulb?")
    submit_button = st.form_submit_with_repricator = st.form_submit_button(label="Submit Question")

# 6. Handle Input Processing
if submit_button and user_query.strip():
    # Save the query
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Historical context instruction setup
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

    # Format transcript history into raw string block
    formatted_prompt = f"{system_instruction}\n\n"
    for msg in st.session_state.messages:
        speaker = "Edison" if msg["role"] == "assistant" else "Student"
        formatted_prompt += f"{speaker}: {msg['content']}\n"
    formatted_prompt += "Edison:"

    with st.spinner("Mr. Edison is writing a reply..."):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=formatted_prompt
            )
            answer = response.text
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()
        except Exception as e:
            st.error(f"AI Connection Error: {e}")
