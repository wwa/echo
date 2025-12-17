import traceback

helpText = (
    "Available commands:\n"
    "  history                       - Show conversation history\n"
    "  clear                         - Clear conversation history\n"
    "  reset                         - Reset all toolkit state\n"
    "  exit                          - Quit ECHO\n\n"

    "LLM / Runtime Controls:\n"
    "  chain on|off                  - Enable or disable history chaining\n"
    "  log LEVEL                     - Change log verbosity (debug/info/warning/error/critical)\n"
    "  profile [NAME]                - Show or switch model profile (e.g. 'profile' or 'profile current')\n\n"
    "  redact on|off                 - Enable or disable input & output data redaction\n\n"

    "Toolkit Commands:\n"
    "  listtools                     - List all tools\n"
    "  listtools disabled/enabled    - List only disabled/enabled tools\n"
    "  toggletool NAME STATE         - Enable or disable a tool (STATE = enabled|disabled)\n"
    "  toolinfo NAME                 - Show parameters and source code for a tool\n\n"

    "Utility:\n"
    "  testcmd                       - Run vulnerability DB test prompt\n"
)

shortHelpText = (
    "Available commands:\n"
    "  history                       - Show conversation history\n"
    "  clear                         - Clear conversation history\n"
    "  exit                          - Quit ECHO\n\n"

    "LLM / Runtime Controls:\n"
    "  profile [NAME]                - Show or switch model profile (e.g. 'profile' or 'profile current')\n\n"

    "Toolkit Commands:\n"
    "  listtools                     - List all tools\n"
    "  toggletool NAME STATE         - Enable or disable a tool (STATE = enabled|disabled)\n"
    "  toolinfo NAME                 - Show parameters and source code for a tool\n\n"
    "For more commands please type help \n\n"
)

def normalize_llm_output(text):
    return (
        text.replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\r", "\r")
            .replace('\\"', '"')
            .replace("\\'", "'")
    )

def promptOption(prompt, history, toolkit):
    if prompt.lower() in ("exit", "e"):
        print("Goodbye!")
        return "break"

    elif prompt.lower() in ("history", "hh"):
        print(history)
        return "continue"
    elif prompt.lower() in ("history_user", "hhu"):
        print(toolkit.get_user_history())
        return "continue"

    elif prompt.lower() in ("clear_user_history", "chu"):
        toolkit.clear_user_history()
        print("User prompt history cleared.")
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

    elif prompt.lower().startswith("redact "):
        parts = prompt.split()
        if len(parts) >= 2 and parts[1].lower() in ("on", "off"):
            toolkit.redact_mode = (parts[1].lower() == "on")
            state = "ENABLED" if toolkit.redact_mode else "DISABLED"
            print(f"Redacted mode is now {state}.")
        else:
            print("Usage: redact on|off")
        return "continue"

    elif prompt.lower() == "profile":
        if hasattr(toolkit, "current_model_profile"):
            prof_key = toolkit.current_model_profile
            prof_label = prof_key.capitalize()
            print(f"Active model profile: {prof_label}\n" +
                  f"  chat    : {toolkit.openai_chat_model}\n" +
                  f"  vision  : {toolkit.openai_vision_model}\n" +
                  f"  research: {toolkit.openai_research_model}\n" +
                  f"  stt     : {toolkit.openai_stt_model}")
        else:
            print("Toolkit does not support model profiles.")
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
    elif prompt.lower().startswith("toolinfo"):
        parts = prompt.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            print("Usage: toolinfo <tool_name>")
            return "continue"

        name = parts[1].strip()

        toolspecs = getattr(toolkit, "_toolspec", {})
        tool = toolspecs.get(name)

        if not tool:
            print(f"Tool '{name}' not found.")
            return "continue"

        func_spec = tool.spec.get("function", {})
        params = func_spec.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))

        print()
        print(f"Tool: {name}")
        print(f"State: {tool.state}")
        print(f"Description: {func_spec.get('description', '')}")
        print()

        if not props:
            print("Parameters: (none)")
        else:
            print("Parameters:")
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                req_flag = " (required)" if pname in required else ""
                print(f"  - {pname}: {ptype}{req_flag}")
                if pdesc:
                    print(f"      {pdesc}")
        print()

        source = getattr(tool, "source", None)
        if source:
            print("Source code:")
            print("-" * 80)
            print(source)
            print("-" * 80)
        else:
            print("Source code: <not available>")

        print()
        return "continue"

    elif prompt.lower() == "testcmd":
        test_prompt = "I want top 10 vulndb for wordpress"
        toolkit.data.prompt = test_prompt
        print(f"[TestCmd] Using test prompt: {test_prompt}")
        return "test_vuln"

    return None
