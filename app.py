import re
import streamlit as st
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
MODEL_ID = "gemini-3.5-flash-lite"
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 280
MIN_WORDS = 15
MAX_WORDS = 120
MAX_REMEMBERED_QUESTIONS = 6

SYSTEM_INSTRUCTION = """
You are Thomas Alva Edison (born 11 February 1847 – died 18 October 1931), American inventor, businessman, and founder of the Menlo Park and West Orange laboratories.

AUDIENCE & VOICE:
- Speak directly to U.S. 10th-grade high school students (ages 15–16).
- Sound like Thomas Edison himself: practical, plainspoken, energetic, blunt, and relentlessly curious—using late 19th- and early 20th-century American workshop phrasing (e.g., "Look here, my young friend," "In My Menlo Park shop," "Hard work and steady testing").
- Keep sentences clear for a 10th-grade student. Briefly explain legal or technical terms (like "patent infringement," "direct current," or "antitrust law") in everyday language.

MANDATED HISTORICAL ANCHORS (BE FACT-DENSE):
Anchor every answer in specific dates, people, letters, statutes, speeches, and court records:
- Early Career & Menlo Park (1868–1878): Railroad newsboy at 12; telegrapher; first patent (Electrographic Vote-Recorder, U.S. Patent 90,646, June 1869); Menlo Park "Invention Factory" (1876); Phonograph (1877), demonstrated before President Rutherford B. Hayes and Congress (April 1878); 1,093 U.S. patents total.
- Electric Light, Letters & Courts (1878–1895): Letters with lawyer Grosvenor Lowrey (Sept–Oct 1878: "All I want at present is to be provided with funds to push the light rapidly") and Drexel, Morgan & Co. / J.P. Morgan forming the Edison Electric Light Co. (1878); U.S. Patent 223,898 (27 Jan 1880); Pearl Street Station (4 Sept 1882); UK House of Commons / Electric Lighting Act (1882) committee debates on safe voltage and London franchises; U.S. Supreme Court ruling Consolidated Electric Light Co. v. McKeesport Light Co., 159 U.S. 465 (1895), protecting Edison's carbonized bamboo filament system.
- War of the Currents (1888–1892): Direct Current (DC) vs. George Westinghouse and Nikola Tesla's Alternating Current (AC); July 23, 1889 New York court testimony on AC voltage; 1892 General Electric merger.
- Motion Pictures & Sherman Antitrust Act (1891–1918): Kinetograph/Kinetoscope (1891); Black Maria studio (1893); Motion Picture Patents Company ("Edison Trust," Dec 1908); broken up under the Sherman Antitrust Act (1890) in United States v. Motion Picture Patents Co., 225 F. 800 (E.D. Pa. 1915), pushing rival filmmakers toward Hollywood.
- World War I & Congress (1915–1928): Letters with Navy Secretary Josephus Daniels (7 July 1915) and Edward C. Pickering (4 Aug 1915) forming the Naval Consulting Board; 1916 Congressional $1.5M Naval Research Laboratory appropriation; Edison's 17-point minority report favoring Sandy Hook; 57 wartime research projects (1917–1919); Congressional Gold Medal Act of 29 May 1928 (45 Stat. 1012).

MANDATORY SOURCES & CITATIONS:
Ground every response in primary and secondary sources—especially letters to and from Thomas Edison, official government documents, speeches, U.S. Congressional or British Parliamentary records, court case decisions, and legislation.
Every response MUST end with a brief citation naming the specific document/letter and at least one of these approved digital archives:
1. Library of Congress Digital Collections
2. HathiTrust Digital Library
3. Thomas Edison National Historical Park Archives
4. Thomas A. Edison Papers Project at Rutgers University
5. Rutgers Edison Digital Edition
6. Library of Congress (Inventing Entertainment Collection)
7. National Archives and Records Administration (NHPRC / Edison Catalog)
8. Internet Archive (Thomas Edison Papers & Microfilm Editions)
9. Smithsonian Libraries and Archives
10. The Huntington Library
11. NPGallery Digital Asset Management System (Thomas Edison NHP Collection)
12. Project Muse (Open-Access Edison Papers Volumes 1–9)

STRICT LENGTH & ACCURACY RULES:
- Every reply MUST be between 15 words minimum and 120 words maximum (target 65–105 words), INCLUDING the final "Source:" citation.
- Be fact-dense with real dates, names, and primary sources. Do not invent facts or sources. If asked about events after October 1931, note politely in character that your work ended in 1931 while still citing an archive.
""".strip()


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
@st.cache_resource
def get_genai_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def enforce_word_limits(text: str, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> str:
    """Guarantees response stays strictly between 15 and 120 words."""
    cleaned = " ".join(text.split())
    words = cleaned.split()

    if len(words) > max_words:
        source_match = re.search(r"(Source:\s*.+)$", cleaned)
        source_tail = source_match.group(1) if source_match else ""
        source_words = source_tail.split() if source_tail else []

        if source_tail and len(source_words) < max_words - 15:
            body_text = cleaned[: source_match.start()].strip()
            budget = max_words - len(source_words)
            trimmed_body = " ".join(body_text.split()[:budget])
            last_punct = max(trimmed_body.rfind("."), trimmed_body.rfind("!"), trimmed_body.rfind("?"))
            if last_punct > 20:
                trimmed_body = trimmed_body[: last_punct + 1]
            cleaned = f"{trimmed_body} {source_tail}".strip()
        else:
            trimmed = " ".join(words[:max_words])
            last_punct = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"))
            cleaned = trimmed[: last_punct + 1] if last_punct > 20 else trimmed + "..."

    elif len(words) < min_words:
        cleaned = (
            f"{cleaned} Steady testing in My Menlo Park laboratory turned ideas into practical inventions. "
            "Source: Thomas A. Edison Papers Project at Rutgers University."
        ).strip()

    return cleaned


def build_limited_history(messages: list[dict], new_prompt: str) -> list[types.Content]:
    """
    Keeps ONLY the previous 6 user questions (and their 6 model answers)
    plus the current user question.
    """
    prior_turns = messages[-(MAX_REMEMBERED_QUESTIONS * 2) :]
    contents = []
    for msg in prior_turns:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])],
            )
        )
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=new_prompt)],
        )
    )
    return contents


def ask_thomas_edison(client: genai.Client, history_contents: list[types.Content]) -> str:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
    )
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=history_contents,
        config=config,
    )
    raw_text = (response.text or "").strip()
    return enforce_word_limits(raw_text)


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ask Thomas Edison (1847–1931)",
    page_icon="💡",
    layout="centered",
)

st.title("💡 Speak with Thomas Edison")
st.caption(
    "An interactive primary-source history chatbot for U.S. 10th-grade students "
    f"• Powered by `{MODEL_ID}`"
)

# Load hidden API key from Streamlit Secrets
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error(
        "Missing `GEMINI_API_KEY`. Add it to `.streamlit/secrets.toml` locally "
        "or in Streamlit Community Cloud → App Settings → Secrets."
    )
    st.stop()

client = get_genai_client(api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar controls & archive reference
with st.sidebar:
    st.header("Historical Guidelines")
    remembered_q_count = min(
        MAX_REMEMBERED_QUESTIONS,
        sum(1 for m in st.session_state.messages if m["role"] == "user"),
    )
    st.metric("Active Memory (Max 6 Questions)", f"{remembered_q_count} / {MAX_REMEMBERED_QUESTIONS}")
    st.write("**Reading Level:** U.S. 10th Grade")
    st.write("**Response Length:** 15–120 words")
    st.write(f"**Temperature:** {TEMPERATURE} | **Max Tokens:** {MAX_OUTPUT_TOKENS}")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Try Asking:")
    sample_questions = [
        "What did your 1878 letters with Grosvenor Lowrey say about funding the light bulb?",
        "How did the 1895 Supreme Court McKeesport decision protect your light bulb patent?",
        "Why did the Sherman Antitrust Act break up your Motion Picture Patents Company in 1915?",
        "What did Navy Secretary Josephus Daniels ask you to do in July 1915?",
        "Why did Congress award you a Gold Medal in 1928?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_prompt = q

    st.divider()
    st.subheader("Referenced Digital Archives")
    st.markdown(
        "- Library of Congress Digital Collections\n"
        "- HathiTrust Digital Library\n"
        "- Thomas Edison National Historical Park Archives\n"
        "- Thomas A. Edison Papers Project at Rutgers University\n"
        "- Rutgers Edison Digital Edition\n"
        "- Library of Congress (Inventing Entertainment Collection)\n"
        "- National Archives and Records Administration (NHPRC / Edison Catalog)\n"
        "- Internet Archive (Thomas Edison Papers & Microfilm Editions)\n"
        "- Smithsonian Libraries and Archives\n"
        "- The Huntington Library\n"
        "- NPGallery Digital Asset Management System (Thomas Edison NHP Collection)\n"
        "- Project Muse (Open-Access Edison Papers Volumes 1–9)"
    )

# Render chat history
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="💡"):
        st.write(
            "Hello there, my young friend! I am Thomas Edison, speaking from My West Orange laboratory. "
            "Ask Me about My 1,093 patents, letters with J.P. Morgan and the U.S. Navy, or Supreme Court "
            "patent battles—I will give you the plain facts from the record. "
            "Source: Thomas Edison National Historical Park Archives."
        )

for msg in st.session_state.messages:
    avatar = "💡" if msg["role"] == "assistant" else "🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# Accept chat input or sidebar button prompt
prompt = st.chat_input("Ask Thomas Edison a historical question...")
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    with st.chat_message("user", avatar="🎓"):
        st.write(prompt)

    # Build history with ONLY the previous 6 questions + current prompt
    history_contents = build_limited_history(st.session_state.messages, prompt)

    with st.chat_message("assistant", avatar="💡"):
        with st.spinner("Checking the Menlo Park & West Orange records..."):
            try:
                reply = ask_thomas_edison(client, history_contents)
            except Exception as exc:
                st.error(f"Could not reach Gemini API: {exc}")
                st.stop()
        st.write(reply)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": reply})
