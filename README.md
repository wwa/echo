# ECHO
ECHO is a modular GPT (4 or newer) based assistant capable of reflection, research and vision.

ECHO stands for Enhanced Computational Heuristic Oracle (don't judge me, it named itself).

It's capable of executing python functions from a toolkit that it has direct control over.
It can enable/disable existing functions or add new ones from source code that it can write itself.
Or use Vision + OCR copy to code from a Twitter post of a screenshot of StackOverflow.

## Requirements

-   Python 3.13.7+

-   `.env` file with:

        OPENAI_API_KEY=your_key_here

-   Optional:

    -   Microphone (for STT)
    -   `WEBSOCKET_LOG_ENABLED=true` for WebSocket log viewer

Install dependencies:

    pip install -r requirements.txt

## How to Run & Install

### 1. Create and activate virtual environment

    python -m venv vEcho
    source vEcho/bin/activate

### 2. Configure environment

    cp .env.example .env

### 3. Start ECHO

Recommended:

    ./run.sh

Manual:

    python Echo.py

## Debug Features

### Logging System

Stored in `logs/`: - important.log - llm.log - trace.log - tools.log -
other.log

### WebSocket Log Viewer

Enable:

    WEBSOCKET_LOG_ENABLED=true

Open `wsClient.html` for live logs.

### CLI Commands

-   help
-   history
-   clear
-   reset
-   log `<level>`
-   chain on/off
-   profile `<name>`
-   listtools
-   toggletool `<name>` enabled\|disabled


# Architecture
LLM Tech stack:
 - Legacy
   - `TTS` Whisper 
   - `Chat` GPT-4 
   - `vision` GPT-4-Vision + Tesseract 
   - GPT-4 Assistants + pyttsx3
   - `research` gpt-4-turbo
 - Modern
   - `TTS` gpt-4o-mini-transcribe
   - `Chat` GPT-5-mini
   - `vision` GPT-5 + Tesseract 
   - `research`GPT-5.1

-  `STT`
    - I prefer ElevenLabs quality, but I daily drive local for lower latency.

GPT is used as a decision-maker to collect data, execute various subordinate functions and present results back to you.

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

## Warnings

ECHO is powerful and can: 
- Execute system-level functions 
- Capture
screenshots and clipboard data
- Be vulnerable to prompt injection Use
only in trusted environments.
