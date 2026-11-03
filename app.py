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

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 4. Handle Student Input
if user_query := st.chat_input("Ask Mr. Edison a question..."):
    # Show student's question immediately
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Historical context prompt for the AI
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

    # Convert past messages into standard text format for the API to avoid version crashes
    formatted_prompt = f"{system_instruction}\n\n"
    for msg in st.session_state.messages:
        speaker = "Edison" if msg["role"] == "assistant" else "Student"
        formatted_prompt += f"{speaker}: {msg['content']}\n"
    formatted_prompt += "Edison:"

    # Call Gemini model
    with st.chat_message("assistant"):
        with st.spinner("Mr. Edison is writing..."):
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=formatted_prompt
                )
                answer = response.text
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()  # Refresh screen to update chat log cleanly
            except Exception as e:
                st.error(f"Error generating response: {e}")
