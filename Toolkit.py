import inspect
import json
import secrets
import traceback
from types import ModuleType
from openai import OpenAI
from dotenv import load_dotenv
from timeit import default_timer as timer
import logging

# Tool imports
import shodan
import time
import os
import webbrowser
import threading
import pytesseract
import pyperclip
import pyttsx3
import base64
import pywinctl as pwc
import pyautogui
import serpapi
import arxiv
import urllib
import urllib.parse
from playsound3 import playsound
import speech_recognition as sr
import mss
import mss.tools
from PIL import ImageGrab, Image
from io import BytesIO
import subprocess
import shlex
import requests
import pyxploitdb

class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self

def genToolspec(name, desc, args={}, reqs=[], **kwargs):
  # openAI tool_calls specification json
  # TODO: validate vs schema
  return {
    'type': 'function',
    'function': {
      'name': name,
      'description': desc,
      "parameters": {
        "type": "object",
        "properties": args,
        "required": reqs
      }
    }
  }
def toolspec(**kwargs):
  def decorator(func):
    if not hasattr(func, '_toolspec'):
      func._toolspec = AttrDict()
    source = kwargs.get('source')
    if source is None:
      try:
        source = inspect.getsource(func)
      except:
        pass
    func._toolspec = AttrDict({
      'state'    : kwargs.get('state',"enabled"),
      'function' : func, 
      'spec'     : genToolspec(name = func.__name__, **kwargs),
      'source'   : source,
      'prompt'   : kwargs.get('prompt',"")
    })
    return func
  return decorator
def b64(img):
  if isinstance(img, Image.Image):
    with BytesIO() as buf:
      img.save(buf, format="PNG")
      return base64.b64encode(buf.getvalue()).decode('utf-8')
  with open(img, "rb") as f:
    return base64.b64encode(f.read()).decode('utf-8')

class Toolkit:
  # Contains toolkit barebones
  def __init__(self):
    self.data             = AttrDict()
    self.module           = ModuleType("DynaToolKit")
    self._toolspec        = AttrDict()
    self.logger           = logging.getLogger(f"echo.toolkit.{self.__class__.__name__}")
    self.trace            = logging.getLogger("echo.trace")
    self.echo_toolkit     = logging.getLogger("echo.toolkit")
    for name in dir(self):
      func = getattr(self, name)
      if not callable(func):
        continue
      if not hasattr(func, '_toolspec'):
        continue
      func._toolspec.function = func
      self._toolspec[name] = func._toolspec

    load_dotenv()

    self.shodan_api_key = os.getenv("SHODAN_API_KEY", "Missing Key")
    self.nvd_api_key = os.getenv("NVD_API_KEY")
    self.enable_listen = os.getenv("ENABLE_LISTEN", "false").lower() == "true"
    self.enable_speak = os.getenv("ENABLE_SPEAK", "false").lower() == "true"

    # --- OpenAI config from .env ---
    self.openai_api_key  = os.getenv("OPENAI_API_KEY")
    self.openai_base_url = os.getenv("OPENAI_BASE_URL")

    # --- Base model values (used for 'current' profile by default) ---
    default_chat     = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini")
    default_vision   = os.getenv("OPENAI_VISION_MODEL", "gpt-5.0")
    default_research = os.getenv("OPENAI_RESEARCH_MODEL", "gpt-5.1")
    default_stt      = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")

    # Store active values (will be overridden by profile)
    self.openai_chat_model     = default_chat
    self.openai_vision_model   = default_vision
    self.openai_research_model = default_research
    self.openai_stt_model      = default_stt

    # --- Named model profiles (template sets) ---
    # You can extend this dict with more profiles later.
    self.model_profiles = {
      "legacy": {
        "chat":     os.getenv("OPENAI_CHAT_MODEL_LEGACY", "gpt-4-turbo-preview"),
        "vision":   os.getenv("OPENAI_VISION_MODEL_LEGACY", "gpt-4-vision-preview"),
        "research": os.getenv("OPENAI_RESEARCH_MODEL_LEGACY", "gpt-4-turbo-preview"),
        "stt":      os.getenv("OPENAI_STT_MODEL_LEGACY", "whisper-1"),
      },
      "current": {
        "chat":     os.getenv("OPENAI_CHAT_MODEL_CURRENT",  default_chat),
        "vision":   os.getenv("OPENAI_VISION_MODEL_CURRENT", default_vision),
        "research": os.getenv("OPENAI_RESEARCH_MODEL_CURRENT", default_research),
        "stt":      os.getenv("OPENAI_STT_MODEL_CURRENT",    default_stt),
      },
    }

    # Select starting profile from env, default 'current'
    self.current_model_profile = os.getenv("OPENAI_MODEL_PROFILE", "current").lower()

    client_kwargs = {}
    if self.openai_api_key:
      client_kwargs["api_key"] = self.openai_api_key
    if self.openai_base_url:
      client_kwargs["base_url"] = self.openai_base_url

    if client_kwargs:
      self.openai = OpenAI(**client_kwargs)
    else:
      self.openai = None

    # Apply selected profile at startup
    try:
      self._apply_model_profile(self.current_model_profile)
    except Exception:
      self.logger.exception("Failed to apply model profile '%s'", self.current_model_profile)


  def reset(self):
    print("Resetting Toolkit")

  def toolspecBySrc(self, src, context=""):
    # Generates openAI tool_calls specifications from source code
    #   WARNING: model-generated, not bulletproof.
    if not self.openai:
      raise Exception("Model-assisted functions unavailable")
    res = self.openai.chat.completions.create(
      model    = self.openai_chat_model,
      messages = [{
        "role": "system",
        "content": f"""
          A Function description is an object describing a function and its arguments
          It consists of 3 elements:
            1. name: function name
            2. description: a short (2 sentences max) description of what the function does.
            3. arguments: an argument description
          An argument description is: {{name:<name>, type:<type>, description: <description>}} where description is a short (2 senteces max) description of the arguments purpose.
          <type> must be one of: number/integer/string
          If function require ApiKey. ApiKey should be compatible with setApiKey tool.
          Generate a function descriptions for each function in source code shown below.
          Answer in JSON {{functions: [{{name:<name>, description:<description>, args=[array of argument description]}},]}}
          <code>
          {src}
          </code>
          <context>
          {context}
          </context>
        """
      }],
      response_format={ "type": "json_object" }
    )
    descs = json.loads(json.loads(res.choices[0].message.model_dump_json())['content'])["functions"]
    tools = []
    for desc in descs:
      args = {}
      reqs = []
      for a in desc['args']:
        # args[a['name']] = {'type':a['type'], 'description':a['type']}
        # forcing type:string because models have weird ideas when generating types (e.g. type:url)
        args[a['name']] = {'type':'string', 'description':a['description']}
        reqs.append(a['name'])
      tools.append(genToolspec(desc['name'],desc['description'],args,reqs))
    return tools
  def addTool(self, func, spec, source=None, prompt=""):
    dec = toolspec(
      desc   = spec['function']['description'],
      args   = spec['function']['parameters']['properties'],
      reqs   = spec['function']['parameters']['required'],
      source = source,
      prompt = prompt
    )
    dec(func)
    self._toolspec[func.__name__] = func._toolspec
    return "{status: success}"
  def addToolByRef(self, func):
    # Registers a function by reference
    src  = inspect.getsource(func)
    spec = self.toolspecBySrc(src)[0]
    return self.addTool(func, spec, src)
  def toolPrompt(self):
    prompt = ""
    for k in self._toolspec:
      tool = self._toolspec[k]
      if tool.state == "enabled":
        prompt += tool.prompt
    return prompt
  def toolMessage(self):
    # Generates tool_calls table
    msgs = []
    for k in self._toolspec:
      tool = self._toolspec[k]
      if tool.state == "enabled":
        msgs.append(tool.spec)
    return msgs
  def call(self, cid, func):
    ts_s = timer()
    self.logger.info("Tool call requested: %s", func.name)
    #self.trace.info("ACTION: LLM selected tool '%s' (tool_call_id=%s) (prop=%s)", func.name, cid, self._toolspec[func.name])
    self.trace.info(
      f"ACTION: LLM selected tool '{func.name}' (tool_call_id={cid}) (args={getattr(func, 'arguments', None)})"
    )

    res = "Error: Unknown error."

    if func.name not in self._toolspec:
      res = "Error: Function not found."
      self.logger.error("Tool %s not found", func.name)
      self.trace.warning("ACTION: Tool '%s' not found", func.name)
    elif self._toolspec[func.name].state == "disabled":
      res = "Error: Function is disabled."
      self.logger.warning("Tool %s is disabled", func.name)
      self.trace.info("ACTION: Tool '%s' is disabled, skipping call", func.name)
    else:
      try:
        args = json.loads(func.arguments)
        self.logger.info("Calling tool %s with args=%s", func.name, args)
        self.trace.info("ACTION: Calling tool '%s' with args=%s", func.name, args)
        self.echo_toolkit.info("Tool %s Output:\n %s", func.name, args)
        res = self._toolspec[func.name].function(**args)
        self.logger.info("Tool %s completed successfully.", func.name)
        self.trace.info("ACTION: Tool '%s' completed.", func.name)
      except Exception as e:
        res = f"Error: <backtrace>\n{traceback.format_exc()}\n</backtrace>"
        self.logger.error("Tool %s raised exception: %s", func.name, e)
        self.trace.error("ACTION: Tool '%s' raised exception: %s", func.name, e)
        print(res)

    ts_e = timer()
    self.logger.info("Tool %s finished in %.3fs", func.name, ts_e - ts_s)
    self.trace.info("ACTION: Tool '%s' finished in %.3fs", func.name, ts_e - ts_s)
    print(f"... took {ts_e-ts_s}s")


    output = {
      "role": "tool",
      "tool_call_id": cid,
      "name": func.name,
      "content": json.dumps({"result": res})
    }

    self.echo_toolkit.info("Tool %s Output:\n %s", func.name, output)

    return output

  def fake(self,name,args='{}'):
    # Fake a tool call. Saves a model call while preserving context flow.
    # Use to pre-emptively inject data into history.
    func = AttrDict({'name':name, 'arguments':args})
    cid  = f"call_{secrets.token_urlsafe(24)}" # mimicking OpenAI IDs. Probably overkill.
    res  = self.call(cid,func)
    return [{
      'role': 'assistant',
      'tool_calls': [{
        'id': cid,
        'function': {
          'arguments': args,
          'name': name
        },
        'type': 'function'
      }],
    }, res]
  
  @toolspec(desc="Lists functions available in toolkit. Lists only disabled function by default.")
  def listTools(self, disabled=True):
    tools = []
    for name in self._toolspec:
      tool = self._toolspec[name]
      if tool.state == 'disabled' or not disabled:
        tools.append({'name': name, 'description': tool.spec['function']['description'], 'state':tool.state})
    return tools
  
  @toolspec(
    desc = "Toggles tool state: enabled/disabled. Disabled tools are not added to tool_calls, saving tokens",
    args = {
      "name":  {"type": "string", "description": "Python source code of functions to be added to toolkit"},
      "state": {"type": "string", "description": "One of: enabled/disabled"}
    },
    reqs = ["name","state"]
  )
  def toggleTool(self, name, state):
    #TODO: check if model thinks history is valid if a tool_call is removed
    if name not in self._toolspec:
      return f"{{status: error, error:{name} not found}}"
    self._toolspec[name].state = state
    return "{status: success}"
  
  @toolspec(
    desc = "Adds functions defined by Python source code to the toolkit. This should only be used if user explicitly asked to add a function to toolkit.",
    args = {"src": {"type": "string", "description": "Python source code of functions to be added to toolkit"}},
    reqs = ["src"]
  )
  def addToolBySrc(self, src):
    # Registers a function by source code
    logs  = ""
    code  = compile(src, self.module.__name__, 'exec')
    specs = self.toolspecBySrc(src)
    exec(code, self.module.__dict__)
    for spec in specs:
      print(spec)
      name = spec['function']['name']
      func = getattr(self.module, name)
      logs += self.addTool(func, spec, src)
    return logs

  def _update_env_var(self, key, value):
    os.environ[key] = value
    env_path = os.path.join(os.getcwd(), ".env")
    try:
      lines = []
      if os.path.exists(env_path):
        with open(env_path, "r") as f:
          lines = f.read().splitlines()

      prefix = key + "="
      found = False
      new_lines = []
      for line in lines:
        if line.startswith(prefix):
          new_lines.append(f"{key}={value}")
          found = True
        else:
          new_lines.append(line)

      if not found:
        new_lines.append(f"{key}={value}")

      with open(env_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

      return True
    except Exception as e:
      print(f"Warning: could not persist {key} to .env: {e}")
      return False

  @toolspec(
    desc="Set or update API keys for external services (OpenAI, Shodan, SerpAPI).",
    args={
      "service": {
        "type": "string",
        "description": "Service name: 'openai', 'shodan', or 'serpapi'."
      },
      "api_key": {
        "type": "string",
        "description": "API key or token for the given service."
      }
    },
    reqs=["service", "api_key"]
  )
  def setApiKey(self, service, api_key):
    svc = service.lower()
    persisted = False

    if svc == "openai":
      # Update OpenAI client
      self.openai_api_key = api_key
      persisted = self._update_env_var("OPENAI_API_KEY", api_key)
      client_kwargs = {"api_key": api_key}
      if getattr(self, "openai_base_url", None):
        client_kwargs["base_url"] = self.openai_base_url
      self.openai = OpenAI(**client_kwargs)

    elif svc == "shodan":
      self.shodan_api_key = api_key
      persisted = self._update_env_var("SHODAN_API_KEY", api_key)
      self.shodan = shodan.Shodan(api_key)

    elif svc == "serpapi":
      persisted = self._update_env_var("SERPAPI_API_KEY", api_key)
      # serpapi.Client can take api_key explicitly
      self.serpapi = serpapi.Client(api_key=api_key)

    else:
      return json.dumps({
        "status": "error",
        "error": f"Unknown service '{service}'. Use one of: openai, shodan, serpapi."
      })

    return json.dumps({
      "status": "success",
      "service": svc,
      "persisted": persisted
    })

  def _apply_model_profile(self, profile_name: str) -> bool:
    """
    Internal helper: apply a named model profile ('legacy', 'current', etc.).
    Updates chat/vision/research/stt model fields.
    """
    profile_key = profile_name.strip().lower()
    if profile_key not in self.model_profiles:
      self.logger.warning("Unknown model profile '%s'", profile_name)
      return False

    prof = self.model_profiles[profile_key]

    # Only set if present in profile
    if "chat" in prof:
      self.openai_chat_model = prof["chat"]
    if "vision" in prof:
      self.openai_vision_model = prof["vision"]
    if "research" in prof:
      self.openai_research_model = prof["research"]
    if "stt" in prof:
      self.openai_stt_model = prof["stt"]

    self.current_model_profile = profile_key
    self.logger.info(
      "Switched model profile to '%s' (chat=%s, vision=%s, research=%s, stt=%s)",
      profile_key,
      self.openai_chat_model,
      self.openai_vision_model,
      self.openai_research_model,
      self.openai_stt_model,
    )
    return True

  @toolspec(
    desc="Switch between predefined LLM model profiles (e.g. 'legacy', 'current'). "
         "Each profile sets chat/vision/research/STT models as a bundle.",
    args={
      "profile": {
        "type": "string",
        "description": "Name of the model profile to activate, e.g. 'legacy' or 'current'."
      }
    },
    reqs=["profile"]
  )
  def setModelProfile(self, profile):
    ok = self._apply_model_profile(profile)
    if not ok:
      return {
        "status": "error",
        "error": f"Unknown profile '{profile}'. Available: {list(self.model_profiles.keys())}"
      }
    return {
      "status": "success",
      "profile": self.current_model_profile,
      "chat_model": self.openai_chat_model,
      "vision_model": self.openai_vision_model,
      "research_model": self.openai_research_model,
      "stt_model": self.openai_stt_model,
    }

  
class BaseToolkit(Toolkit):
  # Contains basic user communication functions
  def __init__(self):
    super(BaseToolkit, self).__init__()
    self.data.stt = None
    self.shodan = shodan.Shodan(self.shodan_api_key)
    self.serpapi = serpapi.Client()
    self.chain_enabled = True
  def stt(self, file=None):
    if file is None:
      file = self.data.stt.file
    with open(file, "rb") as f:
      return self.openai.audio.transcriptions.create(
        model=self.openai_stt_model,
        file=f,
        response_format="text"
      )
  @toolspec(desc="Get input from speech-to-text. Used for primary prompt but can also be called for clarifications/followups/how-to-proceed advice. Category: input, audio", state = "disabled")
  def listen(self):
    if not self.enable_listen:
        return "{status: disabled, reason: 'Speech input disabled in .env'}"
    self.trace.info("ACTION: Starting microphone capture for speech input.")
    if self.data.stt is None:
      rec = sr.Recognizer()
      mic = sr.Microphone()
      self.data.stt = AttrDict({'rec':rec, 'mic':mic, 'file':'./stt.mp3'})
      with mic:
        rec.adjust_for_ambient_noise(mic)
    with self.data.stt.mic:
      audio = self.data.stt.rec.listen(self.data.stt.mic)
    with open(self.data.stt.file, "wb") as f:
      f.write(audio.get_wav_data(convert_rate=44100))
  @toolspec(desc="Get input from console. Used for primary prompt but can also be called for clarifications/followups/how-to-proceed advice. Category: input, text, console")
  def read(self):
    self.trace.info("ACTION: Reading text from console (input()).")
    return input()
  def input(self):
    text = None
    if 'listen' in self._toolspec and self._toolspec.listen.state == "enabled" and self.enable_listen:
      self.trace.info("ACTION: Listening to your voice (speech-to-text).")
      self.listen()
      text = self.stt()
      self.trace.info("ACTION: Transcribed your voice input.")
    else:
      self.trace.info("ACTION: Waiting for your console text input.")
      text = self.read()
      self.trace.info("ACTION: Received your text input from console.")

    self.data.prompt     = text
    self.data.screenshot = None
    self.data.clipboard  = None

    self.trace.info("ACTION: Reading clipboard snapshot.")
    self.clipboardRead()
    self.trace.info("ACTION: Capturing screenshot snapshot.")
    self.screenshot()

    return text
  def userPrompt(self):
    return self.data.prompt

  @toolspec(
    desc = "Open URL in default web browser. Can be a local path with file:/// URL",
    args = {"url": {"type": "string", "description": "URL to be opened"}},
    reqs = ["url"]
  )
  def browse(self, url):
    webbrowser.open(url, new=2)
    return "{status: success}"
  @toolspec(
    desc = "Downloads file from URL. Returns local path of downloaded file.",
    args = {"url": {"type": "string", "description": "File to download"}},
    reqs = ["url"]
  )
  def download(self, url, filename=None):
    # downloads to tmp by default
    file, _ = urllib.request.urlretrieve(url, filename)
    return f"{{status: success, file={file}}}"
  
  @toolspec(
    desc = "Search the Internet. Returns top 10 results: {url, title, description}",
    args = {"phrase": {"type": "string",  "description": "Phrase to search for"},
            "limit":  {"type": "integer", "description": "Number of results. Default: 10"}},
    reqs = ["phrase"],
    state = "disabled",
  )
  def webSearch(self, phrase, limit=10):
    res = self.serpapi.search({'engine': 'google','q': phrase})
    arr = [{'url': r['link'], 'title':r['title'], 'description': r['snippet']} for r in res['organic_results'][:limit]]
    return f"{{status: success, content:{json.dumps(arr)}}}"
  
  def localtts(self,text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    return "{status: success}"
  @toolspec(
    desc = """
      Speak text using text-to-speech. Keep it short and entertaining. Jarvis style banter is welcome.
      Speak should only be used for very short communication - single sentence summary, remark or progress update.
      Category: output, audio
      """,
    args = {"text": {"type": "string", "description": "Text to be spoken. Keep short, one sentence."}},
    reqs = ["text"],
    prompt = "When user says 'say','tell' etc use speak.",
    state = "disabled"
  )
  def speak(self, text):
    if not self.enable_speak:
        return "{status: disabled, reason: 'Speech output disabled in .env'}"
    self.trace.info("ACTION: Speaking short response via TTS.")
    threading.Thread(target=self.localtts, kwargs={'text': text}).start()
    return "{status: success}"

  def screenshot(self, title=None):
    self.trace.info("ACTION: Capturing your screen (primary monitor).")
    with mss.mss() as sct:
      monitor = sct.monitors[1]  # 0 = all, 1 = primary
      img = sct.grab(monitor)
      img_pil = Image.frombytes("RGB", img.size, img.rgb)
      self.data.screenshot = img_pil
    return "{status: success}"

  def selectImage(self, image=None):
    if image is None:
      try:
        image = self.data.clipboard
        if not isinstance(image, Image.Image):
          image = Image.open(image)
      except:
        image = None
    if image is None:
      image = self.data.screenshot
    return image
  
  @toolspec(
    desc = "Optical character recognition to extract text from image. Category: input, image",
    args = {"image": {"type": "string", "description": "Image file to OCR. If not specified, clipboard or screenshot will be used automatically."}},
    reqs = []
  )
  def ocr(self, image=None):
    image = self.selectImage(image)
    return f"{{status: success, content:{pytesseract.image_to_string(image)}}}"
  @toolspec(
    desc = """
      Performs image processing using vision model. 
      Clipboard image or screenshot will be used automatically.
      Category: input, image""",
    args = {"prompt": {"type": "string", "description": "Prompt for vision model. User prompt will also be available for context."}},
    reqs = ["prompt"],
    prompt = "Plan: If clipboard data seems short or not suitable, consider calling vision instead."
  )
  def vision(self, prompt, img=None):
    img = self.selectImage(img)
    ocr = self.ocr(img)
    res = self.openai.chat.completions.create(
      model=self.openai_vision_model,
      max_tokens=500,
      messages=[{
        "role": "system",
        "content": f"""
          You are a subordinate function of an assistant called Echo.
          Echo determined that users request is related to this image and called you.
          You are not talking to the user directly. Be succint. Avoid pleasentries, appologizing and trivial explanations.
          OCR data of the image is provided below. 
          For context the user request to Echo was: {{{self.data.prompt}}}
          If user request is about textual data take a guess on what's important, extract it from OCR and return it verbatim.
          If user request is not about text or if OCR data is not useful to the request, proceed as you see fit yourself.
          <ocr>
          {ocr}
          </ocr>
        """
        },
        {
          "role": "user",
          "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{b64(img)}"}
          ]
        }]
    )
    return res.choices[0].message.content
  @toolspec(
    desc = "Write text into users clipboard. Should be used to output code, json, csv, commands to run, or data to fill a form. Category: output, text, copy-paste",
    args = {"text": {"type": "string", "description": "Text to be written into clipboard"}},
    reqs = ["text"]
  )
  def clipboardWrite(self, text):
    self.trace.info("ACTION: Writing text to your clipboard.")
    pyperclip.copy(text)
    return "{status: success}"
  
  @toolspec(desc="Read contents of users clipboard. Returns {status:<status>, type:<type of content>, content: <text content>}. Category: input, text, copy-paste")
  def clipboardRead(self):
    self.trace.info("ACTION: Attempting to read your clipboard.")
    img = None

    # Try image clipboard first
    try:
      img = ImageGrab.grabclipboard()
    except NotImplementedError as e:
      # Wayland / missing backend (wl-paste/xclip)
      print(f"Image clipboard not supported on this system: {e}")
    except Exception as e:
      print(f"Error grabbing image from clipboard: {e}")

    if isinstance(img, Image.Image):
      self.data.clipboard = img
      return '{"status": "success", "type": "image"}'

    # Fallback to text via pyperclip
    try:
      text = pyperclip.paste()
      self.data.clipboard = text
      # use json.dumps to keep JSON valid even if text has quotes/newlines
      return '{"status": "success", "type": "text", "content": ' + json.dumps(text) + '}'
    except Exception as e:
      print(f"Clipboard text read failed: {e}")
      return '{"status": "error", "reason": "Clipboard not accessible"}'

  @toolspec(
    desc = "Search arxiv for publications. Returns {url:<permalink>, title:<title>, authors:<authors>, summary:<summary>}",
    args = {
      "query":  {"type": "string",  "description": "Arxiv query."},
      "limit":  {"type": "integer", "description": "Optional. Number of results. Default: 10"}
    },
    reqs = ["query"]
  )
  def arxivSearch(self, query, limit=10):
    print(f"{query}")
    client = arxiv.Client()
    res = client.results(arxiv.Search(
      query = query,
      max_results = limit
    ))
    entries = []
    for r in res:
      entries.append({'url': r.entry_id, 'title':r.title, 'authors':r.authors, 'summary':r.summary})
    return f"{{status: success, results:{entries}}}"
  
  @toolspec(
    desc = """ Run a research model. Reseach model can access files and run code.
      Multiple files can be passes in with "files" argument. Supports local files and Arxiv permalinks.
      Pass research_id to continue research. Creates new research thread if empty.
    """,
    args = {
      "query":  {"type": "string", "description": "Research query."},
      "files":  {"type": "array",  "description": "Optional. Array of strings. List of files to include in research. Can be local files or Arxiv permalinks.", "items": {"type": "string"}},
      "research_id":  {"type": "string",  "description": "Optional. Research thread id. If empty, a new research thread will be created."},
    },
    reqs = ["query"],
    prompt = "When researching better results are achieved by reusing existing research thread and uploading multiple files to one thread."
  )
  def research(self, query, files=[], research_id=None):
    ass = None
    thr = None
    if not research_id:
      ass = self.openai.beta.assistants.create(
        instructions="""
          You are a research assistant.
          Your job is to process scientific papers.
          Display mathematical formulas using MathJax \\[ markdown \\] blocks.
        """,
        name  = "Echo research",
        tools = [{"type": "code_interpreter"}, {"type": "retrieval"}],
        model = self.openai_research_model
      )
      thr = self.openai.beta.threads.create(metadata={'aid':ass.id})
      print(f"New research context: {thr.id}")
    else:
      thr = self.openai.beta.threads.retrieve(research_id)
      ass = self.openai.beta.assistants.retrieve(thr.metadata['aid'])
      print(f"Loaded research context: {thr.id}")
    for file in files:
      print(f"Loading file: {file}")
      if not os.path.isfile(file):
        file = urllib.parse.urlparse(file).path.rsplit("/", 1)[-1]
        res  = arxiv.Search(id_list=[file])
        pdf  = next(res.results())
        file = pdf.download_pdf(dirpath="./downloads/")
      with open(file, "rb") as f:
        fid = self.openai.files.create(file = f, purpose = "assistants")
        self.openai.beta.assistants.files.create(assistant_id = ass.id, file_id = fid.id)
    print(f"Research query: {query}")
    ts_s = timer()
    msg  = self.openai.beta.threads.messages.create(thread_id = thr.id, role="user", content = query)
    run  = self.openai.beta.threads.runs.create(assistant_id = ass.id, thread_id = thr.id)
    #time.sleep(5) # FIXME?
    while run.status != "completed":
      time.sleep(1)
      run = self.openai.beta.threads.runs.retrieve(run_id = run.id, thread_id = run.thread_id)
    msg  = self.openai.beta.threads.messages.list(thread_id=run.thread_id,limit=1).data[0].content[0].text.value
    ts_e = timer()
    print(f"... took {ts_e-ts_s}s")
    return {'research_id': thr.id, 'message': msg}

  @toolspec(
    desc="Interact with Shodan API. Search for internet-connected devices.",
    args={
      "query": {"type": "string", "description": "The search query to pass to Shodan."},
    },
    reqs=["query"]
  )
  def shodanSearch(self, query):
    results = self.shodan.search(query)
    return f"{{status: success, results:{results}}}"

  @toolspec(
    desc="Interact with Shodan API. Search for internet-connected devices.",
    args={
      "ip_address": {"type": "string", "optional": True, "description": "IP address to get host information for."}
    },
    reqs=["ip_address"]
  )
  def shodanHostInfo(self, ip_address):
    host_info = self.shodan.host(ip_address)
    return f"{{status: success, result:{host_info}}}"

  @toolspec(
    desc="Change which OpenAI model the toolkit uses at runtime.",
    args={
      "target": {
        "type": "string",
        "description": "Which model to change: one of 'chat', 'vision', 'research', 'stt'."
      },
      "model": {
        "type": "string",
        "description": "New OpenAI model name, e.g. 'gpt-4.1-mini' or 'gpt-4.1'."
      }
    },
    reqs=["target", "model"]
  )
  def setOpenAIModel(self, target, model):
    target_l = target.lower()
    if target_l == "chat":
      self.openai_chat_model = model
    elif target_l == "vision":
      self.openai_vision_model = model
    elif target_l == "research":
      self.openai_research_model = model
    elif target_l == "stt":
      self.openai_stt_model = model
    else:
      return json.dumps({
        "status": "error",
        "error": f"Unknown target '{target}'. Use one of: chat, vision, research, stt."
      })

    return json.dumps({
      "status": "success",
      "target": target_l,
      "model": model
    })

  @toolspec(
    desc="Change logging level at runtime. Useful to increase or decrease verbosity (debug/info/warning/error/critical).",
    args={
      "level": {
        "type": "string",
        "description": "New log level: one of 'debug', 'info', 'warning', 'error', 'critical'."
      },
      "logger_name": {
        "type": "string",
        "description": "Optional logger name, default 'echo'. For fine-grained control you can use 'echo.llm' or 'echo.toolkit.BaseToolkit'."
      }
    },
    reqs=["level"]
  )
  def setLogLevel(self, level, logger_name="echo"):
    lvl_str = level.upper()
    mapping = {
      "DEBUG": logging.DEBUG,
      "INFO": logging.INFO,
      "WARNING": logging.WARNING,
      "ERROR": logging.ERROR,
      "CRITICAL": logging.CRITICAL,
    }
    if lvl_str not in mapping:
      return {
        "status": "error",
        "error": f"Unknown level '{level}'. Use one of: debug, info, warning, error, critical."
      }

    lvl = mapping[lvl_str]
    logger = logging.getLogger(logger_name)
    logger.setLevel(lvl)

    # Optionally: also raise root level so everything respects it
    if logger_name == "echo":
      logging.getLogger().setLevel(lvl)

    return {
      "status": "success",
      "logger": logger_name,
      "level": lvl_str
    }
  @toolspec(
    desc=(
      "Search Exploit-DB (exploit-db.com) for exploits by keyword or CVE. "
      "Returns a list of exploits with id, description, type, platform, date, "
      "verified flag, port, tags, author, and link. Uses pyxploitdb."
    ),
    args={
      "query": {
        "type": "string",
        "description": "Search term (product, version, or CVE like 'CVE-2021-44228')."
      },
      "limit": {
        "type": "integer",
        "description": "Maximum number of results to return (default 10)."
      }
    },
    reqs=["query"]
  )
  def exploitdbSearch(self, query, limit=10):
    if pyxploitdb is None:
      return json.dumps({
        "status": "error",
        "error": "pyxploitdb not installed; run `pip install pyxploitdb`"
      })

    try:
      # ✅ Correct call for this library: searchEDB (camelCase)
      results = pyxploitdb.searchEDB(query, _print=False, nb_results=limit)

      payload = []
      for e in results:
        payload.append({
          "id": getattr(e, "id", None),
          "description": getattr(e, "description", None),
          "type": getattr(e, "type", None),
          "platform": getattr(e, "platform", None),
          "date_published": getattr(e, "date_published", None),
          "verified": getattr(e, "verified", None),
          "port": getattr(e, "port", None),
          "tags": getattr(e, "tag_if_any", None),
          "author": getattr(e, "author", None),
          "link": getattr(e, "link", None),
        })

      return json.dumps({
        "status": "success",
        "query": query,
        "results": payload
      })
    except Exception as ex:
      return json.dumps({
        "status": "error",
        "error": f"Exploit-DB search failed: {ex}"
      })

  @toolspec(
    desc=(
      "Search Exploit-DB specifically by CVE identifier "
      "(e.g. 'CVE-2006-1234', 'CVE-2021-44228'). "
      "Uses pyxploitdb.searchCVE under the hood."
    ),
    args={
      "cve": {
        "type": "string",
        "description": "CVE identifier to search for."
      },
      "limit": {
        "type": "integer",
        "description": "Maximum number of results (default 10)."
      }
    },
    reqs=["cve"]
  )
  def exploitdbSearchCVE(self, cve, limit=10):
    if pyxploitdb is None:
      return json.dumps({
        "status": "error",
        "error": "pyxploitdb not installed; run `pip install pyxploitdb`"
      })

    try:
      results = pyxploitdb.searchCVE(cve, _print=False)
      # searchCVE may return more than we want; truncate
      results = results[:limit]

      payload = []
      for e in results:
        payload.append({
          "id": getattr(e, "id", None),
          "description": getattr(e, "description", None),
          "type": getattr(e, "type", None),
          "platform": getattr(e, "platform", None),
          "date_published": getattr(e, "date_published", None),
          "verified": getattr(e, "verified", None),
          "port": getattr(e, "port", None),
          "tags": getattr(e, "tag_if_any", None),
          "author": getattr(e, "author", None),
          "link": getattr(e, "link", None),
        })

      return json.dumps({
        "status": "success",
        "cve": cve,
        "results": payload
      })
    except Exception as ex:
      return json.dumps({
        "status": "error",
        "error": f"Exploit-DB CVE search failed: {ex}"
      })
  @toolspec(
    desc=(
      "Query the U.S. National Vulnerability Database (NVD) for vulnerabilities. "
      "Supports keyword, CVE ID, product name, vendor name, etc. "
      "Requires NVD_API_KEY in .env. Returns normalized list of CVE records."
    ),
    args={
      "query": {
        "type": "string",
        "description": "Search string or CVE ID (e.g., 'Apache', 'OpenSSL', 'CVE-2021-44228')."
      },
      "limit": {
        "type": "integer",
        "description": "Maximum number of results to return (default 10)."
      }
    },
    reqs=["query"]
  )
  def nvdSearch(self, query, limit=10):
    """
    Search NVD (National Vulnerability Database) using API v2.
    """
    api_key = self.nvd_api_key
    if not api_key:
      return json.dumps({
        "status": "error",
        "error": "NVD_API_KEY not set in .env"
      })

    base = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    params = {
      "keywordSearch": query,
      "resultsPerPage": limit,
      "apiKey": api_key
    }

    try:
      res = requests.get(base, params=params, timeout=12)
      res.raise_for_status()
      data = res.json()

      out = []
      for v in data.get("vulnerabilities", []):
        cve = v.get("cve", {})

        out.append({
          "id": cve.get("id"),
          "published": cve.get("published"),
          "lastModified": cve.get("lastModified"),
          "description": self._extractNvdDescription(cve),
          "cvss": self._extractNvdCvss(cve),
          "weaknesses": self._extractNvdWeaknesses(cve),
          "references": self._extractNvdReferences(cve)
        })

      return json.dumps({
        "status": "success",
        "query": query,
        "results": out
      })

    except Exception as ex:
      return json.dumps({
        "status": "error",
        "error": f"NVD query failed: {ex}"
      })


  # ---- helper methods -------
  def _extractNvdDescription(self, cve):
    descs = cve.get("descriptions", [])
    for d in descs:
      if d.get("lang") == "en":
        return d.get("value")
    return None

  def _extractNvdCvss(self, cve):
    metrics = cve.get("metrics", {})
    if "cvssMetricV31" in metrics:
      item = metrics["cvssMetricV31"][0]
      return {
        "baseScore": item["cvssData"]["baseScore"],
        "vector": item["cvssData"]["vectorString"]
      }
    if "cvssMetricV30" in metrics:
      item = metrics["cvssMetricV30"][0]
      return {
        "baseScore": item["cvssData"]["baseScore"],
        "vector": item["cvssData"]["vectorString"]
      }
    if "cvssMetricV2" in metrics:
      item = metrics["cvssMetricV2"][0]
      return {
        "baseScore": item["cvssData"]["baseScore"],
        "vector": item["cvssData"]["vectorString"]
      }
    return None

  def _extractNvdWeaknesses(self, cve):
    entries = []
    for w in cve.get("weaknesses", []):
      for desc in w.get("description", []):
        entries.append(desc.get("value"))
    return entries

  def _extractNvdReferences(self, cve):
    refs = []
    for r in cve.get("references", []):
      refs.append({
        "url": r.get("url"),
        "source": r.get("source"),
        "tags": r.get("tags")
      })
    return refs
  @toolspec(
    desc = """
      Install software using a system package manager.
      Supported managers: apt, apt-get, gem, pip, pip3, npm, dnf, yum, pacman.
      Use ONLY when the user explicitly asks to install something.
      """,
    args = {
      "manager": {
        "type": "string",
        "description": "Package manager to use: apt, apt-get, gem, pip, pip3, npm, dnf, yum, pacman."
      },
      "packages": {
        "type": "string",
        "description": "One or more package names, separated by spaces (e.g. 'curl git')."
      },
      "options": {
        "type": "string",
        "description": "Optional extra flags, e.g. '-y' or '--version \\'1.2.3\\''."
      }
    },
    reqs = ["manager", "packages"]
  )
  def installSoftware(self, manager, packages, options=""):
    """
    Install software packages using a system package manager.

    SECURITY NOTE:
    - This runs commands on the host system.
    - Only enable/use this if you trust the model+prompt, and the environment is yours.
    """
    logger = logging.getLogger("echo.toolkit.install")

    manager = manager.strip()
    options = options or ""

    # Whitelist allowed managers
    allowed = {"apt", "apt-get", "gem", "pip", "pip3", "npm", "dnf", "yum", "pacman"}
    if manager not in allowed:
      return {
        "status": "error",
        "error": f"Manager '{manager}' not allowed. Allowed: {sorted(list(allowed))}"
      }

    # Build base command
    cmd = [manager]

    # For these managers, auto-add "install" subcommand if not provided
    auto_install = {"apt", "apt-get", "dnf", "yum", "pacman"}
    if manager in auto_install:
      cmd.append("install")

    # Add extra options if given
    if options.strip():
      try:
        cmd.extend(shlex.split(options))
      except ValueError as e:
        return {
          "status": "error",
          "error": f"Could not parse options: {e}"
        }

    # Split packages by whitespace into a list
    pkg_list = [p for p in packages.split() if p.strip()]
    if not pkg_list:
      return {
        "status": "error",
        "error": "No packages specified after splitting 'packages' by spaces."
      }

    cmd.extend(pkg_list)

    logger.info("Running install command: %s", cmd)

    try:
      proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True
      )
    except FileNotFoundError:
      return {
        "status": "error",
        "error": f"Command '{manager}' not found on this system."
      }
    except Exception as e:
      logger.exception("installSoftware raised exception")
      return {
        "status": "error",
        "error": f"Exception while running install: {e}"
      }

    result = {
      "status": "success" if proc.returncode == 0 else "error",
      "returncode": proc.returncode,
      "command": cmd,
      "stdout": proc.stdout,
      "stderr": proc.stderr
    }

    # Log a short summary
    if proc.returncode == 0:
      logger.info("Install succeeded: %s", cmd)
    else:
      logger.warning("Install failed (rc=%s): %s", proc.returncode, cmd)

    return result

  @toolspec(
    desc = "Enable or disable remembering previous conversation turns when answering. When disabled, only the latest user prompt is sent to the model.",
    args = {
      "enabled": {
        "type": "string",
        "description": "Set to 'true' to enable history chaining, or 'false' to disable it."
      }
    },
    reqs = ["enabled"]
  )
  def setHistoryChain(self, enabled):
    """
    Toggle whether the assistant uses previous conversation turns as context.
    """
    val = enabled.strip().lower()
    on = val in ("true", "1", "yes", "y", "on")

    self.chain_enabled = on

    return {
      "status": "success",
      "chain_enabled": self.chain_enabled
    }