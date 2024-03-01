import inspect
import json
import secrets
import traceback
from types import ModuleType
from openai import OpenAI
from dotenv import load_dotenv
from timeit import default_timer as timer

# Tool imports
import time
import os
import webbrowser
import threading
import pytesseract
import clipboard
import pyttsx3
import base64
import pygetwindow
import pyautogui
import serpapi
import arxiv
import urllib
import urllib.parse
from playsound import playsound
import speech_recognition as sr
from PIL import ImageGrab, Image
from io import BytesIO

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
      'state'    : 'enabled',
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
    self.data      = AttrDict()
    self.module    = ModuleType("DynaToolKit")
    self._toolspec = AttrDict()
    for name in dir(self):
      func = getattr(self, name)
      if not callable(func):
        continue
      if not hasattr(func, '_toolspec'):
        continue
      func._toolspec.function = func # overwrite with bound ref
      self._toolspec[name] = func._toolspec
    load_dotenv()
    if "OPENAI_API_KEY" in os.environ:
      self.openai = OpenAI()
    else:
      # model-assisted functions like addToolBySrc will be unavailable
      self.openai = None
  def toolspecBySrc(self, src, context=""):
    # Generates openAI tool_calls specifications from source code
    #   WARNING: model-generated, not bulletproof.
    if not self.openai:
      raise Exception("Model-assisted functions unavailable")
    res = self.openai.chat.completions.create(
      model    = "gpt-4-turbo-preview",
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
    # Calls a tool.
    #   func is a message.tool_calls[i].function object
    ts_s = timer()
    print(f"Calling {func.name}")
    res = "Error: Unknown error."
    if func.name not in self._toolspec:
      res = "Error: Function not found."
    elif self._toolspec[func.name].state == "enabled":
      res = "Error: Function is disabled."
    try:
      args = json.loads(func.arguments)
      res = self._toolspec[func.name].function(**args)
    except Exception as e:
      # very important! most of the time model will correct itself if you let it know where it screwed up.
      res = f"Error: <backtrace>\n{traceback.format_exc()}\n</backtrace>"
      print(res)
      pass
    ts_e = timer()
    print(f"... took {ts_e-ts_s}s")
    return {
      "role": "tool", 
      "tool_call_id": cid,
      "name": func.name, 
      "content": f'{{"result": {str(res)}}}'
    }
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
  
class BaseToolkit(Toolkit):
  # Contains basic user communication functions
  def __init__(self):
    super(BaseToolkit, self).__init__()
    self.data.stt = None
    self.serpapi  = serpapi.Client()
  def stt(self, file=None):
    if file is None:
      file = self.data.stt.file
    with open(file, "rb") as f:
      return self.openai.audio.transcriptions.create(model="whisper-1", file=f, response_format="text")
  @toolspec(desc="Get input from speech-to-text. Used for primary prompt but can also be called for clarifications/followups/how-to-proceed advice. Category: input, audio")
  def listen(self):
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
    return input()
  def input(self):
    text = None
    if 'listen' in self._toolspec and self._toolspec.listen.state == "enabled":
      self.listen()
      text = self.stt()
    else:
      text = self.read()
    self.data.prompt     = text
    self.data.screenshot = None
    self.data.clipboard  = None
    # gather clipboard and screenshot at the time of prompt
    # tool calls can take a moment and screen/clipboard can change in the meantime
    self.clipboardRead()
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
  def download(url, filename=None):
    # downloads to tmp by default
    file, _ = urllib.request.urlretrieve(url, filename)
    return f"{{status: success, file={file}}}"
  
  @toolspec(
    desc = "Search the Internet. Returns top 10 results: {url, title, description}",
    args = {"phrase": {"type": "string",  "description": "Phrase to search for"},
            "limit":  {"type": "integer", "description": "Number of results. Default: 10"}},
    reqs = ["phrase"]
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
    prompt = "When user says 'say','tell' etc use speak."
  )
  def speak(self, text):
    threading.Thread(target=self.localtts, kwargs={'text':text}).start()
    return "{status: success}"
  
  def screenshot(self,title=None):
    win = pygetwindow.getActiveWindow()
    if title:
      win = pygetwindow.getWindowsWithTitle(title)[0]
    img = None
    if win:
      img = pyautogui.screenshot(region=(win.left, win.top, win.width, win.height))
    else:
      img = pyautogui.screenshot()
    self.data.screenshot = img
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
      model="gpt-4-vision-preview",
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
    clipboard.copy(text)
    return "{status: success}"
  
  @toolspec(desc="Read contents of users clipboard. Returns {status:<status>, type:<type of content>, content: <text content>}. Category: input, text, copy-paste")
  def clipboardRead(self):
    img = ImageGrab.grabclipboard()
    if img:
      self.data.clipboard = img
      return f"{{status: success, type: image}}"
    self.data.clipboard = clipboard.paste()
    return f"{{status: success, type: text, content:{self.data.clipboard}}}"

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
          Display mathematical formulas using MathJax \[ markdown \] blocks.
        """,
        name  = "Echo research",
        tools = [{"type": "code_interpreter"}, {"type": "retrieval"}],
        model = "gpt-4-turbo-preview"
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

