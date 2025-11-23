import traceback

def promptOption(prompt, history, helpText, toolkit):
    if prompt.lower() in ("exit", "e"):
        print("Goodbye!")
        return "break"

    elif prompt.lower() in ("history", "hh"):
        print(history)
        return "continue"

    elif prompt.lower() in ("clear", "c"):
        history.clear()
        return "continue"

    elif prompt.lower() in ("reset", "r"):
        toolkit.reset()
        print("All tools reset.")
        return "continue"

    elif prompt.lower() in ("help", "h", "?"):
        print(helpText)
        return "continue"

    elif prompt.lower().startswith("log "):
        parts = prompt.split()
        if len(parts) >= 2:
            level_name = parts[1].upper()
            if level_name in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
                logger = toolkit.logger
                import logging
                logger.setLevel(getattr(logging, level_name))
                print(f"Log level changed to {level_name}")
            else:
                print("Unknown log level. Use: debug, info, warning, error, critical.")
        else:
            print("Usage: log <level>")
        return "continue"

    elif prompt.lower().startswith("chain "):
        parts = prompt.split()
        if len(parts) >= 2 and parts[1].lower() in ("on", "off"):
            toolkit.chain_enabled = (parts[1].lower() == "on")
            print(f"Conversation history chaining is now {'ENABLED' if toolkit.chain_enabled else 'DISABLED'}.")
        else:
            print("Usage: chain on|off")
        return "continue"

    elif prompt.lower().startswith("profile "):
        parts = prompt.split()
        profile = parts[1] if len(parts) >= 2 else ""
        if hasattr(toolkit, "_apply_model_profile"):
            ok = toolkit._apply_model_profile(profile)
            if ok:
                print(f"Model profile switched to '{profile}'.")
                print(f"  chat    : {toolkit.openai_chat_model}")
                print(f"  vision  : {toolkit.openai_vision_model}")
                print(f"  research: {toolkit.openai_research_model}")
                print(f"  stt     : {toolkit.openai_stt_model}")
            else:
                print(f"Unknown profile '{profile}'. Available: {', '.join(toolkit.model_profiles.keys())}")
        else:
            print("Toolkit does not support model profiles.")
        return "continue"

    elif prompt.lower() == "testcmd":
        test_prompt = "I want top 10 vulndb for wordpress"
        toolkit.data.prompt = test_prompt
        print(f"[TestCmd] Using test prompt: {test_prompt}")
        return "test_vuln"

    return None
