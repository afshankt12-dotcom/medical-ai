import streamlit as st
from src.chatbot import MedicalChatbot

import base64
import time
import re
import os
import tempfile

from langdetect import detect
from deep_translator import GoogleTranslator


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Medical AI",
    page_icon="🩺",
    layout="wide"
)


# --------------------------------------------------
# IMAGE TO BASE64
# --------------------------------------------------

def get_base64(path):

    with open(path, "rb") as img:

        return base64.b64encode(
            img.read()
        ).decode()


# --------------------------------------------------
# LOADING SCREEN
# --------------------------------------------------

def loading_screen():

    logo = get_base64("assets/logo.png")

    loader = st.empty()

    loader.markdown(
f"""
<style>

@keyframes spin {{
0% {{
transform: rotate(0deg);
}}

100% {{
transform: rotate(360deg);
}}
}}

.loading {{

height:90vh;

display:flex;

justify-content:center;

align-items:center;

flex-direction:column;

}}

.logo {{

width:180px;

animation:spin 2s linear infinite;

}}

.loading-text {{

margin-top:25px;

font-size:32px;

font-weight:bold;

color:#C62828;

}}

</style>

<div class="loading">

<img
class="logo"
src="data:image/png;base64,{logo}">

<div class="loading-text">

Loading Medical AI...

</div>

</div>

""",
unsafe_allow_html=True
    )

    time.sleep(8)

    loader.empty()


if "loaded" not in st.session_state:

    loading_screen()

    st.session_state.loaded = True


# --------------------------------------------------
# BACKGROUND
# --------------------------------------------------

background = get_base64(
    "assets/background.png"
)
# --------------------------------------------------
# CSS
# --------------------------------------------------

st.markdown(
f"""
<style>

.stApp {{

background-image:url(
"data:image/png;base64,{background}"
);

background-size:cover;

background-position:center;

background-attachment:fixed;

}}

.main .block-container {{

max-width:950px;

margin:auto;

padding:30px;

background:rgba(255,255,255,.90);

border-radius:20px;

box-shadow:0 6px 20px rgba(0,0,0,.25);

}}

.stTextInput input {{

height:55px !important;

border-radius:30px !important;

font-size:18px !important;

padding-left:20px !important;

background:white !important;

color:black !important;

border:2px solid #E53935 !important;

}}

.stButton button {{

width:50px;

height:50px;

border-radius:50%;

font-size:22px;

background:#E53935;

color:white !important;

}}

.user-box {{

background:#E53935;

padding:15px;

border-radius:15px;

margin-top:15px;

color:white !important;

}}

.user-box * {{

color:white !important;

}}

.bot-box {{

background:white;

padding:15px;

border-radius:15px;

border-left:6px solid #E53935;

box-shadow:0 4px 12px rgba(0,0,0,.2);

color:black !important;

}}

.bot-box * {{

color:black !important;

}}

p,
span,
label,
div {{

color:black !important;

}}

footer {{

visibility:hidden;

}}

div[data-testid="stAudioInput"],
div[data-testid="stAudioInput"] > div,
div[data-testid="stAudioInput"] div {{
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

div[data-testid="stAudioInput"] button {{
    background-color: #E53935 !important;
    border-radius: 50% !important;
    width: 50px !important;
    height: 50px !important;
    min-width: 50px !important;
    min-height: 50px !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: transform 0.2s !important;
}}

div[data-testid="stAudioInput"] button:hover {{
    transform: scale(1.05) !important;
    background-color: #D32F2F !important;
}}

div[data-testid="stAudioInput"] button svg {{
    display: none !important;
}}

div[data-testid="stAudioInput"] button::before {{
    content: "" !important;
    display: inline-block !important;
    width: 24px !important;
    height: 24px !important;
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z"/></svg>') !important;
    background-size: contain !important;
    background-repeat: no-repeat !important;
}}

div[data-testid="stAudioInput"] canvas,
div[data-testid="stAudioInput"] audio {{
    display: none !important;
}}

div[data-testid="stAudioInput"] p {{
    display: none !important;
}}

div[data-testid="stAudioInput"] span {{
    color: transparent !important;
    font-size: 0px !important;
    line-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}}
</style>
""",
unsafe_allow_html=True
)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_chatbot():

    return MedicalChatbot()


@st.cache_resource
def load_whisper():

    from faster_whisper import WhisperModel

    return WhisperModel("base", device="cpu", compute_type="int8")


chatbot = load_chatbot()
whisper = load_whisper()
# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.image(
    "assets/girl.png",
    width=120
)

st.markdown(
"""
<h1 style="
text-align:center;
color:#C62828;
font-size:48px;
margin-bottom:10px;
">
🩺 Medical AI
</h1>

<p style="
text-align:center;
font-size:20px;
color:black;
margin-bottom:30px;
">
Ask medical questions and get AI-powered answers.
</p>
""",
unsafe_allow_html=True
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

if "voice_question" not in st.session_state:
    st.session_state.voice_question = ""

if "last_processed_audio" not in st.session_state:
    st.session_state.last_processed_audio = None

if "debug_info" not in st.session_state:
    st.session_state.debug_info = None


# --------------------------------------------------
# CLEAR CHAT BUTTON
# --------------------------------------------------

col1, col2, col3 = st.columns([1,7,1])

with col1:

    if st.button(
        "🗑️",
        use_container_width=True,
        key="clear"
    ):

        st.session_state.messages = []
        st.session_state.voice_question = ""
        st.session_state.last_processed_audio = None
        st.session_state.debug_info = None
        st.rerun()


st.write("")


# --------------------------------------------------
# INPUT BAR
# --------------------------------------------------

input_col, mic_col = st.columns([9, 1])

with input_col:

    text_question = st.text_input(

        "",

        value=st.session_state.voice_question,

        placeholder="Ask anything...",

        label_visibility="collapsed",

        key="question_input"

    )

with mic_col:

    audio_file = st.audio_input(

        "Record voice",

        label_visibility="collapsed",

        key="voice_input"

    )


# --------------------------------------------------
# VOICE INPUT PROCESSING
# --------------------------------------------------

if audio_file is not None:

    audio_id = getattr(audio_file, "id", None) or f"{audio_file.name}_{audio_file.size}"

    if st.session_state.last_processed_audio != audio_id:

        st.session_state.last_processed_audio = audio_id

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:

            temp_file.write(audio_file.read())

            temp_path = temp_file.name


        try:

            with st.spinner("🎤 Transcribing voice..."):

                # Transcribe using cached whisper model
                segments, info = whisper.transcribe(
                    temp_path,
                    beam_size=5
                )

                spoken_text = "".join([segment.text for segment in segments]).strip()


            if spoken_text:

                st.session_state.voice_question = spoken_text

                st.rerun()


        except Exception as e:

            st.error(f"Failed to record or transcribe audio: {e}")


        finally:

            if os.path.exists(temp_path):

                try:

                    os.remove(temp_path)

                except Exception:

                    pass


# --------------------------------------------------
# GET QUESTION
# --------------------------------------------------

question = ""


if st.session_state.voice_question.strip():

    question = st.session_state.voice_question


elif text_question.strip():

    question = text_question



# --------------------------------------------------
# LANGUAGE DETECTION FUNCTION
# --------------------------------------------------

def detect_language(text):

    manglish_keywords = ["enikku", "enik", "enikk", "pani", "und", "undu", "vedana", "chum", "chuma", "vayar"]
    text_lower = text.lower()

    try:

        # Fix English short sentence detection problem / Check for Manglish
        if re.fullmatch(
            r"[A-Za-z0-9 ,.!?'\-()]+",
            text
        ):

            for word in manglish_keywords:
                if word in text_lower:
                    return "ml"

            return "en"


        language = detect(text)


        return language


    except Exception:

        return "en"


# --------------------------------------------------
# MANGLISH CONVERTER
# --------------------------------------------------

def convert_manglish_to_malayalam(text):

    replacements = {
        "vayar vedana": "വയറുവേദന",
        "vayarvedana": "വയറുവേദന",
        "thala vedana": "തലവേദന",
        "thalavedana": "തലവേദന",
        "enikku": "എനിക്ക്",
        "enik": "എനിക്ക്",
        "enikk": "എനിക്ക്",
        "pani": "പനി",
        "undu": "ഉണ്ട്",
        "und": "ഉണ്ട്",
        "chum": "ചുമ",
        "chuma": "ചുമ",
        "vayar": "വയർ",
        "vedana": "വേദന",
        "alla": "അല്ല"
    }

    text_lower = text.lower()

    for phrase in ["vayar vedana", "thala vedana", "vayarvedana", "thalavedana"]:
        if phrase in text_lower:
            text_lower = text_lower.replace(phrase, replacements[phrase])

    words = text_lower.split()
    result = []

    for word in words:
        if word in replacements:
            result.append(replacements[word])
        else:
            result.append(word)

    return " ".join(result)


# --------------------------------------------------
# CHATBOT (AUTO LANGUAGE)
# --------------------------------------------------

if question.strip():

    original_input = question
    user_language = detect_language(question)

    manglish_converted = None
    if user_language == "ml":
        manglish_converted = convert_manglish_to_malayalam(question)
        trans_source = manglish_converted
    else:
        trans_source = question

    # Translate question to English
    if user_language != "en":

        try:

            english_question = GoogleTranslator(
                source="auto",
                target="en"
            ).translate(trans_source)

        except Exception:

            english_question = trans_source

    else:

        english_question = question



    # --------------------------------------------------
    # AI THINKING LOADING
    # --------------------------------------------------

    with st.spinner(
        "🤖 Medical AI is thinking..."
    ):

        time.sleep(2)

        result = chatbot.ask(
            english_question
        )



    raw_answer = result["answer"]
    confidence = result["confidence"]


    # Confidence fallback check (threshold 0.25)
    if confidence < 0.25:
        processed_answer = "I'm not confident enough to answer that question. Please rephrase it or consult a healthcare professional."
    else:
        processed_answer = raw_answer


    # Translate answer back to user language
    if user_language != "en":

        try:

            final_answer = GoogleTranslator(
                source="en",
                target=user_language
            ).translate(processed_answer)

        except Exception:

            final_answer = processed_answer

    else:

        final_answer = processed_answer



    # Save to debug info
    st.session_state.debug_info = {
        "original_input": original_input,
        "detected_language": user_language,
        "manglish_converted": manglish_converted,
        "english_translation": english_question,
        "raw_answer": raw_answer,
        "confidence": confidence,
        "final_answer": final_answer
    }


    # Save message
    st.session_state.messages.append(
        {
            "question": original_input,

            "answer": final_answer,

            "confidence": confidence
        }
    )

    # Clear voice input
    st.session_state.voice_question = ""



# --------------------------------------------------
# CHAT DISPLAY
# --------------------------------------------------

for chat in reversed(
    st.session_state.messages
):

    st.markdown(
        f"""
<div class="user-box">

<b>👤 You</b>

<br><br>

{chat["question"]}

</div>
""",
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
<div class="bot-box">

<b>🤖 Medical AI</b>

<br><br>

{chat["answer"]}

</div>
""",
        unsafe_allow_html=True
    )


    # Confidence

    confidence = float(
        chat.get(
            "confidence",
            0.0
        )
    )


    # Keep value between 0 and 1

    confidence = max(
        0.0,
        min(
            confidence,
            1.0
        )
    )


    st.progress(
        confidence
    )


    st.caption(
        f"Confidence: {confidence * 100:.1f}%"
    )



# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()


st.markdown(
"""
<div style="
text-align:center;
color:black;
font-size:15px;
padding:10px;
">

© 2026 Medical AI

</div>
""",
unsafe_allow_html=True
)