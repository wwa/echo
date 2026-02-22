import traceback
import json
import os
from datetime import datetime

helpText = (
    "Available commands:\n"
    "  history                       - Show conversation history\n"
    "  export [FILENAME]             - Export command history to file (default: echo_history.txt)\n"
    "  export_session [FILENAME]     - Export current session prompts (default: echo_session.txt)\n"
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

    "Sequences:\n"
    "  sequence                      - List all available command sequences\n"
    "  sequence list                 - List all available command sequences\n"
    "  sequence <NAME>               - Execute a command sequence (with step-by-step authorization)\n\n"

    "Utility:\n"
    "  testcmd                       - Run vulnerability DB test prompt\n"
)

shortHelpText = (
    "Available commands:\n"
    "  history                       - Show conversation history\n"
    "  export [FILENAME]             - Export command history to file\n"
    "  export_session [FILENAME]     - Export current session prompts\n"
    "  clear                         - Clear conversation history\n"
    "  exit                          - Quit ECHO\n\n"

    "LLM / Runtime Controls:\n"
    "  profile [NAME]                - Show or switch model profile (e.g. 'profile' or 'profile current')\n\n"

    "Toolkit Commands:\n"
    "  listtools                     - List all tools\n"
    "  toggletool NAME STATE         - Enable or disable a tool (STATE = enabled|disabled)\n"
    "  toolinfo NAME                 - Show parameters and source code for a tool\n\n"

    "Sequences:\n"
    "  sequence                      - List all command sequences\n"
    "  sequence <NAME>               - Execute a command sequence\n\n"

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

def load_sequences():
    """Load command sequences from sequences.json file"""
    try:
        seq_file = os.path.join(os.path.dirname(__file__), "sequences.json")
        if os.path.exists(seq_file):
            with open(seq_file, "r") as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading sequences: {e}")
        return []

def display_sequences():
    """Display all available sequences in a formatted table"""
    sequences = load_sequences()
    if not sequences:
        print("No sequences available.")
        return

    print("\n" + "=" * 80)
    print("Available Command Sequences".center(80))
    print("=" * 80)

    for seq in sequences:
        name = seq.get("name", "unnamed")
        desc = seq.get("desc", "No description")
        cmds = seq.get("cmds", [])
        auto_exec = seq.get("autoExecute", False)

        auto_label = "🤖 AUTO" if auto_exec else "👤 MANUAL"

        print(f"\n📋 {name} [{auto_label}]")
        print(f"   {desc}")
        print(f"   Steps: {len(cmds)}")
        print("   " + "-" * 76)
        for i, cmd in enumerate(cmds, 1):
            print(f"   {i}. {cmd}")

    print("\n" + "=" * 80 + "\n")

def execute_sequence(seq_name, toolkit, history, start_from=1, show_prev_completed=False):
    """Execute a command sequence with step-by-step authorization"""
    sequences = load_sequences()
    sequence = next((s for s in sequences if s.get("name") == seq_name), None)

    if not sequence:
        print(f"⚠️  WARNING: Sequence '{seq_name}' not found.")
        return "continue"

    cmds = sequence.get("cmds", [])
    if not cmds:
        print("⚠️  WARNING: Sequence has no commands.")
        return "continue"

    auto_execute = sequence.get("autoExecute", False)
    mode_label = "🤖 AUTO MODE" if auto_execute else "👤 MANUAL MODE"

    if start_from == 1:
        print(f"\n🚀 Starting sequence: {sequence.get('name')} [{mode_label}]")
        print(f"   {sequence.get('desc', 'No description')}")
        print(f"   Total steps: {len(cmds)}\n")

    for i in range(start_from - 1, len(cmds)):
        step_num = i + 1
        cmd = cmds[i]
        is_last = (step_num == len(cmds))
        step_label = "LAST STEP" if is_last else f"Step {step_num}/{len(cmds)}"

        print(f"\n{'='*60}")
        if show_prev_completed and i == start_from - 1:
            print(f"⚠️  Step {step_num - 1}/{len(cmds)} completed.")
            print(f"{'='*60}")
        print(f"⚠️  {step_label}: {cmd}")
        if not auto_execute:
            print(f"⚠️  Press ENTER to continue, or type anything to stop sequence.")
        print(f"{'='*60}")

        if not auto_execute:
            response = input(">> ").strip()

            if response:
                print("\n⚠️  WARNING: Sequence cancelled by user.")
                return "continue"

        # Execute the command
        result = promptOption(cmd, history, toolkit)

        if result == "break":
            print("\n⚠️  WARNING: Sequence terminated.")
            return "break"
        elif result is None:
            # It's a user prompt that needs LLM processing
            print(f"\n▶️  Executing with LLM: {cmd}")
            return ("sequence_exec", seq_name, cmd, step_num, len(cmds), auto_execute)

        # In auto mode, continue automatically
        if auto_execute:
            if is_last:
                print(f"\n✅ Final step completed!")
            else:
                print(f"\n✓ Step {step_num}/{len(cmds)} completed, continuing...")

    print(f"\n✅ Sequence '{seq_name}' completed successfully!\n")
    return "continue"

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

    elif prompt.lower().startswith("export_session"):
        parts = prompt.split(maxsplit=1)
        filename = parts[1].strip() if len(parts) > 1 else "echo_session.txt"

        try:
            session_history = toolkit.get_user_history()
            
            if not session_history:
                print("No prompts in current session.")
                return "continue"
            
            dest_path = os.path.join(os.getcwd(), filename)
            
            with open(dest_path, 'w', encoding='utf-8') as f:
                f.write(f"# ECHO Session Export\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Total prompts: {len(session_history)}\n\n")
                
                for idx, prompt in enumerate(session_history, 1):
                    f.write(f"[{idx}] {prompt}\n")
            
            print(f"Session history exported to: {dest_path}")
            print(f"Total prompts exported: {len(session_history)}")
        except Exception as e:
            print(f"Failed to export session history: {e}")
            traceback.print_exc()

        return "continue"

    elif prompt.lower().startswith("export"):
        parts = prompt.split(maxsplit=1)
        filename = parts[1].strip() if len(parts) > 1 else "echo_history.txt"

        # Get readline history file path
        histfile = os.path.join(os.path.expanduser("~"), ".echo_history")

        try:
            if os.path.exists(histfile):
                # Copy history file to current directory
                import shutil
                dest_path = os.path.join(os.getcwd(), filename)
                shutil.copy2(histfile, dest_path)
                print(f"Command history exported to: {dest_path}")
            else:
                print(f"No history file found at: {histfile}")
        except Exception as e:
            print(f"Failed to export history: {e}")
            traceback.print_exc()

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
                  f"  chat    : {toolkit.chat_model}\n" +
                  f"  vision  : {toolkit.vision_model}\n" +
                  f"  research: {toolkit.research_model}\n" +
                  f"  stt     : {toolkit.stt_model}")
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
                print(f"  chat    : {toolkit.chat_model}")
                print(f"  vision  : {toolkit.vision_model}")
                print(f"  research: {toolkit.research_model}")
                print(f"  stt     : {toolkit.stt_model}")
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

    elif prompt.lower() == "sequence" or prompt.lower() == "sequence list":
        display_sequences()
        return "continue"

    elif prompt.lower().startswith("sequence "):
        parts = prompt.split(maxsplit=1)
        if len(parts) >= 2:
            seq_name = parts[1].strip()
            if seq_name.lower() == "list":
                display_sequences()
                return "continue"
            else:
                return execute_sequence(seq_name, toolkit, history)
        else:
            display_sequences()
            return "continue"

    return None
