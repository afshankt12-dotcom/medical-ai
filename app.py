import streamlit as st
from src.chatbot import MedicalChatbot
from src.voice import VoiceRecognizer

import base64
import time
import re

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
def load_voice():

    return VoiceRecognizer()


chatbot = load_chatbot()

voice = load_voice()
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

if "voice_question" not in st.session_state:
    st.session_state.voice_question = ""

if "processing" not in st.session_state:
    st.session_state.processing = False


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
        st.rerun()


st.write("")


# --------------------------------------------------
# INPUT BAR
# --------------------------------------------------

input_col, mic_col = st.columns([9,1])

with input_col:

    text_question = st.text_input(

        "",

        placeholder="Ask anything...",

        label_visibility="collapsed",

        key="question_input"

    )

with mic_col:

    voice_button = st.button(

        "🎤",

        key="voice_button",

        use_container_width=True

    )
# --------------------------------------------------
# VOICE INPUT
# --------------------------------------------------

if voice_button:

    try:

        with st.spinner("🎤 Listening..."):

            audio = voice.record_audio(
                duration=5
            )

            text = voice.speech_to_text(
                audio
            )


        if text:

            st.session_state.voice_question = text

            st.rerun()

        else:

            st.warning(
                "No speech detected."
            )


    except Exception as e:

        st.error(
            f"Voice Error: {e}"
        )


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

    try:

        # Fix English short sentence detection problem
        if re.fullmatch(
            r"[A-Za-z0-9 ,.!?'\-()]+",
            text
        ):

            return "en"


        language = detect(text)


        return language


    except Exception:

        return "en"
# --------------------------------------------------
# CHATBOT (AUTO LANGUAGE)
# --------------------------------------------------

if question.strip():

    # Detect user language safely
    try:

        # English text check
        if re.fullmatch(
            r"[A-Za-z0-9 ,.!?'\-()]+",
            question
        ):
            user_language = "en"

        else:
            user_language = detect(question)

    except Exception:

        user_language = "en"



    # Translate question to English
    if user_language != "en":

        try:

            english_question = GoogleTranslator(
                source="auto",
                target="en"
            ).translate(question)

        except Exception:

            english_question = question

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



    answer = result["answer"]



    # Translate answer back to user language
    if user_language != "en":

        try:

            answer = GoogleTranslator(
                source="en",
                target=user_language
            ).translate(answer)

        except Exception:

            pass



    # Save message

    st.session_state.messages.append(
        {
            "question": question,

            "answer": answer,

            "confidence": result["confidence"]
        }
    )


    # Clear voice input

    st.session_state.voice_question = ""# --------------------------------------------------
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