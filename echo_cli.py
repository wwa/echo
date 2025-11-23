import traceback

helpText = (
    "Available commands:\n"
    "  history                - Show conversation history\n"
    "  clear                  - Clear conversation history\n"
    "  reset                  - Reset all toolkit state\n"
    "  exit                   - Quit ECHO\n\n"

    "LLM / Runtime Controls:\n"
    "  chain on|off           - Enable or disable history chaining\n"
    "  log LEVEL              - Change log verbosity (debug/info/warning/error/critical)\n"
    "  profile NAME           - Switch model profile (e.g. 'profile current')\n\n"

    "Toolkit Commands:\n"
    "  listtools              - List all tools\n"
    "  listtools disabled     - List only disabled tools\n"
    "  listtools enabled      - List only enabled tools\n"
    "  toggletool NAME STATE  - Enable or disable a tool (STATE = enabled|disabled)\n\n"

    "Utility:\n"
    "  testcmd                - Run vulnerability DB test prompt\n"
)

def promptOption(prompt, history, toolkit):
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
    elif prompt.lower().startswith("listtools"):
        parts = prompt.split()
        mode = parts[1].lower() if len(parts) > 1 else "all"

        if mode == "disabled":
            tools = toolkit.listTools()
        elif mode == "enabled":
            tools = toolkit.listTools(mode="enabled")
        else:
            tools = toolkit.listTools(mode="all")

        print("\nToolkit Tools:")
        print(mode)
        for t in tools:
            print(f" - {t['name']}  [{t['state']}]")
        print()
        return "continue"


    elif prompt.lower().startswith("toggletool"):
        parts = prompt.split()
        if len(parts) < 3:
            print("Usage: toggletool <name> <enabled|disabled>")
            return "continue"

        name = parts[1]
        state = parts[2].lower()

        if state not in ("enabled", "disabled"):
            print("State must be: enabled or disabled")
            return "continue"

        res = toolkit.toggleTool(name, state)
        print(f"Tool '{name}' set to state '{state}'. Result: {res}")
        return "continue"

    elif prompt.lower() == "testcmd":
        test_prompt = "I want top 10 vulndb for wordpress"
        toolkit.data.prompt = test_prompt
        print(f"[TestCmd] Using test prompt: {test_prompt}")
        return "test_vuln"

    return None
