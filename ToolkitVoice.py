import os
import threading

#Import(own)
from Toolkit import toolspec, AttrDict

ENABLE_LISTEN = os.getenv("ENABLE_LISTEN", "false").lower() == "true"
ENABLE_SPEAK = os.getenv("ENABLE_SPEAK", "false").lower() == "true"

# Import heavy audio libs only if at least one audio feature is enabled
if ENABLE_LISTEN or ENABLE_SPEAK:
    import pyttsx3
    from playsound3 import playsound
    import speech_recognition as sr
else:
    # If completely disabled you can leave these undefined
    pyttsx3 = None
    playsound = None
    sr = None

from Toolkit import BaseToolkit

class BaseToolkitVoice(BaseToolkit):
  def __init__(self):
    super().__init__()

  @toolspec(
    desc="""
        Speak text using text-to-speech. Keep it short and entertaining. Jarvis style banter is welcome.
        Speak should only be used for very short communication - single sentence summary, remark or progress update.
        Category: output, audio
        """,
    args={"text": {"type": "string", "description": "Text to be spoken. Keep short, one sentence."}},
    reqs=["text"],
    prompt="When user says 'say','tell' etc use speak.",
    state="disabled"
  )
  def speak(self, text):
    self.trace.info("ACTION: Speaking short response via TTS.")
    threading.Thread(target=self.localtts, kwargs={'text': text}).start()
    return "{status: success}"

  def stt(self, file=None):
    if file is None:
      file = self.data.stt.file
    with open(file, "rb") as f:
      return self.openai.audio.transcriptions.create(
        model=self.openai_stt_model,
        file=f,
        response_format="text"
      )

  def localtts(self, text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    return "{status: success}"

  @toolspec(
    desc="Get input from speech-to-text. Used for primary prompt but can also be called for clarifications/followups/how-to-proceed advice. Category: input, audio",
    args={},
    reqs=[]
  )
  def listen(self):
    self.trace.info("ACTION: Starting microphone capture for speech input.")
    if self.data.stt is None:
      rec = sr.Recognizer()
      mic = sr.Microphone()
      self.data.stt = AttrDict({'rec': rec, 'mic': mic, 'file': './stt.mp3'})
      with mic:
        rec.adjust_for_ambient_noise(mic)
    with self.data.stt.mic:
      audio = self.data.stt.rec.listen(self.data.stt.mic)
    with open(self.data.stt.file, "wb") as f:
      f.write(audio.get_wav_data(convert_rate=44100))