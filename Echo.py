import os
import json
import traceback
from timeit import default_timer as timer
from toolkit import BaseToolkit

def modelOne(toolkit, messages):
  ts_s = timer()
  print("Prompting...")
  res = toolkit.openai.chat.completions.create(
    model    = "gpt-4-turbo-preview",
    messages = messages,
    tools    = toolkit.toolMessage(),
    tool_choice = "auto"
  )
  ts_e = timer()
  print(f"... took {ts_e-ts_s}s")
  reason  = res.choices[0].finish_reason
  message = res.choices[0].message
  if reason == "stop":
    messages.append(json.loads(message.model_dump_json(exclude={'function_call', 'tool_calls'})))
    return reason, message.content, messages
  if reason == "tool_calls":
    # exclude because model_dump_json produces string Nones which can't be injested back
    messages.append(json.loads(message.model_dump_json(exclude={'function_call', 'content'})))
    for tc in message.tool_calls:
      if tc.type == "function":
        messages.append(toolkit.call(tc.id, tc.function))
  return reason,None,messages
def modelLoop(toolkit, history=[]):
  messages = [{
    "role": "system",
    "content": f"""
      You are a helpful assistant called ECHO.
      Based on user request and available functions devise a plan of action and execute it.
      Keep in mind multiple data sources are available. If you are unable to fulfil the request with one data source, try again with another.
      A demonstrative pronoun such as this/that/these/it likely refers to something in conversation history, or data copied to cliboard or something that user sees on his screen.
      Regardless of action taken, respond in JSON with {{plan:<plan>,response:<text response>}}
    """ + toolkit.toolPrompt()
    }] + sum(history, []) + [{"role":"user", "content":toolkit.userPrompt()}] + toolkit.fake('listTools') + toolkit.fake('clipboardRead')
  content = None
  while True:
    reason, content, messages = modelOne(toolkit, messages)
    if reason == "stop":
      break
  history.append(messages)
  return content, history

def mainLoop(toolkit, limit=10):
  history = []
  while True:
    try:
      prompt = toolkit.input()
      print(prompt)
      content, history = modelLoop(toolkit, history)
      history = history[:limit]
      print(content)
    except Exception as e:
      traceback.print_exc()
      pass

if __name__ == "__main__":
  toolkit = BaseToolkit()
  if not toolkit.openai:
    raise Exception('OpenAI API not initialized')
  # Turn audio off for console I/O:
  # toolkit.toggleTool('listen','disabled')
  # toolkit.toggleTool('speak','disabled')
  mainLoop(toolkit)
