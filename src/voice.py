import tempfile
import scipy.io.wavfile as wav

try:
    import sounddevice as sd
    SOUND_AVAILABLE = True
except Exception:
    sd = None
    SOUND_AVAILABLE = False

from faster_whisper import WhisperModel


class VoiceRecognizer:

    def __init__(self):
        print("Loading Voice Model...")

        self.model = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8"
        )

        print("Voice Model Ready!")

    def record_audio(self, duration=5, sample_rate=16000):

        if not SOUND_AVAILABLE:
            raise RuntimeError(
                "Voice recording is not supported on Streamlit Cloud."
            )

        print("Recording... Speak now")

        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1
        )

        sd.wait()

        file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )

        wav.write(
            file.name,
            sample_rate,
            audio
        )

        return file.name

    def speech_to_text(self, audio_file):

        segments, info = self.model.transcribe(
            audio_file,
            beam_size=5
        )

        text = ""

        for segment in segments:
            text += segment.text

        return text.strip()