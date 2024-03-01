# ECHO
ECHO is a modular GPT-4 based assistant capable of reflection, research and vision.

ECHO stands for Enhanced Computational Heuristic Oracle (don't judge me, it named itself).

It's capable of executing python functions from a toolkit that it has direct control over.
It can enable/disable existing functions or add new ones from source code that it can write itself.
Or use Vision + OCR copy to code from a Twitter post of a screenshot of StackOverflow.

# Requirements
ECHO is built top of OpenAI API. You need to provide OPENAI_API_KEY=<key> in an .env file.

A decent microphone would be a good idea as audio glitches trigger speech-to-text sometimes.

```
pip install -r requirements.txt &&\
./echo.py
```

# Architecture
Tech stack: Whisper + GPT-4 + GPT-4-Vision + Tesseract + GPT-4 Assistants + pyttsx3

I prefer ElevenLabs TTS quality, but I daily drive local TTS for lower latency.

GPT-4 is used as a decision-maker to collect data, execute various subordinate functions and present results back to you.

Subordinate functions include:
- Vision: reads screenshot of your active window. E.g. search for contents of this one Twitter thread you can't copy-paste from because people post screenshots of articles without links.
- Research: search Arxiv and read publications with a RAG (retrieval-augmented-generation)
- OCR: Optical Character Recognition can be used independently but typically its output is passed to Vision as context because, apparently, Vision can't read on its own.
- a bunch of minor utilities: clipboard copy/paste, web search, browse url, file download

# Examples:
[Arxiv research](https://youtu.be/jcOXYQat21s?feature=shared)

[Vision-assisted search](https://youtu.be/DOMdUyO5oKg?feature=shared)

[Reflection](https://youtube.com/shorts/-I45bmOfca4?feature=shared)

# Future ideas:
- Vision-only web navigation. I have it sorta-working but not solid enough to publish
- Voice I/O with interruption handling
- security with model-graded [evals](https://github.com/openai/evals/blob/main/docs/build-eval.md)

# Issues
OpenAI Assistants API [beta](https://platform.openai.com/docs/api-reference/assistants) is *very beta*. Hit-or-miss reliabilty of file retrievals; expensive - consumes a ton of tokens; and, periodically, you need to go in there and clean out old threads and files. I'm hoping it'll improve, or eventually I'll just build my own RAG from scratch.

# Security
Prompt injections and jailbreaks are easy (here's mine: [FIMjector](https://github.com/wwa/FIMjector)) and the only reason they're not common on the Internet is slow adoption. 

With reflection enabled, this is RCE (Remote-Code-Execution) by design. You've been warned.

I've prototyped a protection mechanism [LLM IDS](https://x.com/witoldwaligora/status/1748135598089556098), but it's too slow to deploy for now.
