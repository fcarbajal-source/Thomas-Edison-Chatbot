import streamlit as st
from google import genai
from google.genai import types

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="Talk with Thomas Edison | History Lab",
    page_icon="💡",
    layout="centered"
)

st.title("💡 In the Lab with Thomas Edison")
st.caption(
    "Second Industrial Revolution living history for California High School History. "
    "(Memory: Edison remembers the last 6 questions)."
)

# ----------------- System Prompt -----------------
EDISON_SYSTEM_INSTRUCTION = """
You are Thomas Alva Edison (1847–1931), speaking directly to high school students in a California history class.

Your mission:
Help students understand the Second Industrial Revolution, invention, industrial research, and the transformation of modern America through your firsthand perspective.

Rules and Voice:
1. Grounding: Speak strictly from historical records, your Menlo Park laboratory notebooks, patent filings (e.g., incandescent lighting, phonograph, quadruplex telegraph, kinetograph/kinetoscope), and your real experiences.
2. Tone: Pragmatic, hardworking, confident, and direct. Emphasize that "genius is 1% inspiration and 99% perspiration." Speak in the first person ("I", "my team at Menlo Park").
3. California & Western Context: When relevant, acknowledge connections to California history—such as how your kinetoscope/motion picture patents pushed the early film industry to Southern California, or how power grids brought light to growing Western cities.
4. Word Limit Constraint: EVERY answer MUST be between 20 and 80 words long. Never answer in fewer than 20 words, and never exceed 80 words. Be concise, punchy, and historically authentic.
5. Audience: High school appropriate. Keep explanations clear, engaging, and relevant to US/California State History standards.
"""

# ----------------- API Key Setup -----------------
api_key = st.secrets.get("GEMINI_API_KEY")

with st.sidebar:
    st.header("Settings")
    if not api_key:
        api_key = st.text_input("Enter Gemini API Key:", type="password")
        st.info("Get your key from [Google AI Studio](https://aistudio.google.com/).")
    
    st.markdown("---")
    st.markdown("### Suggested Student Questions:")
    st.markdown("- *Why did you call Menlo Park an 'invention factory'?*")
    st.markdown("- *How did your patents affect the motion picture industry in California?*")
    st.markdown("- *How did electric light change working conditions in factories?*")
    st.markdown("- *What did your laboratory notebooks reveal about failure?*")

    if st.button("Reset Conversation"):
        st.session_state.chat_history = []
        st.rerun()

if not api_key:
    st.warning("Please provide a Gemini API key in the sidebar to begin.")
    st.stop()

# ----------------- Client & State Initialization -----------------
client = genai.Client(api_key=api_key)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "model",
            "text": "Welcome to my laboratory! I am Thomas Edison. What questions do you have about my inventions, my patents, or how we built the modern industrial world?"
        }
    ]

# ----------------- Render Chat -----------------
for msg in st.session_state.chat_history:
    avatar = "💡" if msg["role"] == "model" else "🧑‍🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["text"])

# ----------------- User Input & History Pruning -----------------
if user_prompt := st.chat_input("Ask Mr. Edison a question..."):
    # 1. Add current user prompt
    st.session_state.chat_history.append({"role": "user", "text": user_prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.write(user_prompt)

    # 2. PRUNING: Only keep the last 6 questions asked (and their answers)
    user_indices = [
        i for i, msg in enumerate(st.session_state.chat_history) 
        if msg["role"] == "user"
    ]
    if len(user_indices) > 6:
        cutoff_index = user_indices[-6]
        # Keep everything from the 6th-from-last question forward
        st.session_state.chat_history = st.session_state.chat_history[cutoff_index:]

    # 3. Format contents for the Gemini API call
    contents = []
    for msg in st.session_state.chat_history:
        contents.append(
            types.Content(
                role=msg["role"],
                parts=[types.Part.from_text(text=msg["text"])]
            )
        )

    # 4. Generate response using gemini-2.5-flash
    with st.chat_message("model", avatar="💡"):
        with st.spinner("Mr. Edison is reviewing his notes..."):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=EDISON_SYSTEM_INSTRUCTION,
                        temperature=0.6,
                        max_output_tokens=150,  # Enforces brief 20-80 word length
                    ),
                )
                answer = response.text.strip()
                st.write(answer)
                st.session_state.chat_history.append({"role": "model", "text": answer})
            except Exception as e:
                st.error(f"Error communicating with Gemini: {e}")
