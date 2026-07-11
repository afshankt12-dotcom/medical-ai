import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile

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


    def record_audio(
        self,
        duration=5,
        sample_rate=16000
    ):

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



    def speech_to_text(
        self,
        audio_file
    ):

        segments, info = self.model.transcribe(
            audio_file,
            beam_size=5
        )


        text = ""

        for segment in segments:
            text += segment.text


        return text.strip()