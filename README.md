# ECHO
ECHO is a modular GPT (4 or newer) based assistant capable of reflection, research and vision.

## Demo

![Echo in Action Demo](assets/demo.gif)

## Intro

ECHO stands for Enhanced Computatcional Heuristic Oracle (don't judge me, it named itself).

It's capable of executing python functions from a toolkit that it has direct control over.
It can enable/disable existing functions or add new ones from source code that it can write itself.
Or use Vision + OCR copy to code from a Twitter post of a screenshot of StackOverflow.

## Requirements

- Python 3.14.2+ (Higher versions may require requirements adjustment)

-   `.env` file with:

        OPENAI_API_KEY=your_key_here

-   Optional:

    -   Speakers (for TTS)
    -   Microphone (for STT)
    -   `WEBSOCKET_LOG_ENABLED=true` for WebSocket log viewer

Install dependencies:

    pip install -r requirements.txt

## How to Run & Install

### Option 1: Classic Installation

#### 1. Create and activate virtual environment

    python -m venv vEcho
    source vEcho/bin/activate

#### 2. Install dependencies

    pip install -r requirements.txt

#### 3. Configure environment

    cp .env.example .env

Edit `.env` file and configure your API keys and providers (see Configuration section below).

#### 4. Start ECHO

Recommended:

    ./run.sh

Manual:

    python Echo.py

### Option 2: Docker Installation

#### 1. Configure environment

    cp .env.example .env

Edit `.env` file and configure your API keys and providers (see Configuration section below).

#### 2. Build and run with Docker Compose

    cd docker
    docker-compose up --build

The container will automatically start with:
- Virtual display (Xvfb) for visual perception features
- Persistent volumes for data and logs
- Interactive TTY for CLI interaction

#### 3. Stop the container

    docker-compose down

## Configuration

### Configuring Providers

ECHO supports multiple LLM providers (OpenAI, Anthropic, Google, DeepSeek, etc.). Providers are configured in the `.env` file using the `LLM_PROVIDERS` JSON array.

Each provider entry requires:
- `providerName`: Unique identifier (e.g., "openai", "anthropic")
- `endpoint`: API endpoint URL
- `apiKey`: Your API authentication key
- `desc`: Description of the provider
- `name`: Display name for the UI

Example provider configuration:

```json
LLM_PROVIDERS='[
  {
    "providerName": "openai",
    "endpoint": "https://api.openai.com/v1/",
    "apiKey": "your-api-key-here",
    "desc": "OpenAI Official API",
    "name": "OpenAI"
  },
  {
    "providerName": "anthropic",
    "endpoint": "https://api.anthropic.com/v1/",
    "apiKey": "your-api-key-here",
    "desc": "Anthropic Claude API",
    "name": "Anthropic"
  }
]'
```

### Configuring Models

Models are mapped to providers using the `MODEL_PROVIDER_MAP` in `.env`. This allows using the same model from different providers with unique identifiers.

Each model entry requires:
- `modelName`: Actual model name used in API calls
- `providerName`: Must match a provider from `LLM_PROVIDERS`
- `modelIdentifier`: Unique alias for referencing the model

Optional fields:
- `supportsToolChoice`: Set to `false` if model doesn't support automatic tool calling
- `assistantsToolChoiceOverride`: Override tool_choice for Assistants API ("none", "auto", "required")
- `assistantsFallbackModel`: Use another model for tool routing decisions

Example model configuration:

```json
MODEL_PROVIDER_MAP='[
  {
    "modelName": "gpt-5-mini",
    "providerName": "openai",
    "modelIdentifier": "openai-gpt5-mini"
  },
  {
    "modelName": "claude-3-5-sonnet-20241022",
    "providerName": "anthropic",
    "modelIdentifier": "claude-sonnet-3.5"
  }
]'
```

### Configuring Model Profiles

Model profiles define which models to use for different tasks (chat, vision, research, STT). Profiles are configured in `profiles.json`.

Each profile contains:
- `name`: Profile identifier
- `desc`: Description
- `isLegacy`: Whether this is a legacy profile
- `models`: Object mapping task types to model identifiers

Example profile:

```json
{
  "name": "current",
  "desc": "Use latest models from various providers",
  "isLegacy": false,
  "models": {
    "chat": "openai-gpt5-mini",
    "vision": "openai-gpt5",
    "research": "openai-gpt5.2",
    "stt": "openai-gpt4o-mini-transcribe"
  }
}
```

Set the active profile in `.env`:

    MODEL_PROFILE=current

Switch profiles at runtime using CLI:

    profile <name>

### Configuring Sequences

Sequences allow you to automate command execution. They are configured in `sequences.json`.

Each sequence contains:
- `name`: Sequence identifier
- `desc`: Description of what the sequence does
- `autoExecute`: Whether to execute automatically (true/false)
- `cmds`: Array of commands to execute in order

Example sequence:

```json
{
  "name": "example",
  "desc": "Example vulnerability scan sequence",
  "autoExecute": false,
  "cmds": [
    "profile current",
    "listtools enabled",
    "testcmd"
  ]
}
```

Execute a sequence using CLI:

    sequence <name>

Or execute directly:

    execseq <name>

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
-   toggletool `<name>` enabled|disabled
-   toolinfo `<name>`               Show parameters and source code for a toolkit function

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
   - `vision` GPT-5.2 + Tesseract 
   - `research`GPT-5.2

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
