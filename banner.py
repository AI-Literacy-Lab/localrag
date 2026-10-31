"""On-screen banner for the video: Ollama status + the two model badges. No extra packages needed."""

import os
import urllib.request

os.system("")  # enables ANSI colors in older Windows consoles

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
DIM = "\033[2m"
CYAN_BADGE = "\033[1;30;46m"     # black text on cyan
MAGENTA_BADGE = "\033[1;30;45m"  # black text on magenta


def ollama_running():
    try:
        with urllib.request.urlopen("http://localhost:11434", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def badge(model, color):
    return f"{color} {model} {RESET}"


def show_model_badges():
    if ollama_running():
        print(f"\n{GREEN}{BOLD}● Ollama active{RESET}  {DIM}(localhost:11434){RESET}\n")
    else:
        print(f"\n{RED}{BOLD}● Ollama not running{RESET}  {DIM}(start it with: ollama serve){RESET}\n")

    print(f"  {badge('nomic-embed-text', CYAN_BADGE)}  {BOLD}The Reader{RESET}  {DIM}turns text into vectors{RESET}")
    print(f"  {badge('llama3.2', MAGENTA_BADGE)}  {BOLD}The Writer{RESET}  {DIM}writes the answer from the chunks{RESET}\n")


if __name__ == "__main__":
    show_model_badges()