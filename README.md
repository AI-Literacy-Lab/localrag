
# Local RAG Demo (Ollama + Chroma + LangChain)

A fully local Retrieval-Augmented Generation demo. It loads a document, splits it into chunks, embeds them with `nomic-embed-text`, stores them in Chroma, and answers questions with `llama3.2`. Nothing leaves your machine.

The sample document is a made-up HR handbook for a fictional company (NovaWave Technologies), so you can verify that answers come from retrieval and not from the model's training data.

## Project structure

```
.
├── rag.py              # the demo script
├── sample.txt          # the document to query (HR handbook)
├── requirements.txt    # Python dependencies
└── README.md
```

`chroma_db/` is created automatically when you run the script and is rebuilt on every run.

## Prerequisites

- **Python 3.13**
- **Ollama**
- About 3 GB of free disk space for the models

## 1. Install Python 3.13

### Windows
1. Download Python 3.13 from https://www.python.org/downloads/
2. Run the installer and **tick "Add python.exe to PATH"** before clicking Install.
3. Open a new terminal and verify:
   ```
   python --version
   ```
   Expected: `Python 3.13.x`

If you have several Python versions installed, use the launcher instead:
```
py -3.13 --version
```

### macOS
```
brew install python@3.13
python3.13 --version
```

### Linux (Ubuntu/Debian)
```
sudo apt update
sudo apt install python3.13 python3.13-venv
python3.13 --version
```

## 2. Install Ollama and pull the models

1. Install Ollama from https://ollama.com/download
2. Pull the two models used by the demo:
   ```
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```
3. Check they are available:
   ```
   ollama list
   ```

Ollama must be running in the background when you start the script. The desktop app does this automatically. Otherwise run `ollama serve` in a separate terminal.

## 3. Set up the project

Create and activate a virtual environment, then install the dependencies.

### Windows (PowerShell)
```
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this once and try again:
```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### macOS / Linux
```
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Run the demo

```
python rag.py
```

The script will:
1. Load `sample.txt`
2. Split it into chunks
3. Print the embedding dimension and the first 5 values of one vector
4. Build the Chroma vector store
5. Start an interactive question loop

### Commands in the loop

| Input | What it does |
|-------|--------------|
| `<question>` | RAG answer. Shows the retrieved chunks and distance scores, then the answer |
| `!<question>` | Plain LLM answer with no retrieval (to show hallucination) |
| `exit` or `quit` | Leave the program |

### Suggested demo questions

1. `!How many days of annual leave do I get?` : the plain LLM guesses
2. `How many days of annual leave do I get?` : correct answer is 27
3. `What's the notice period for an L6?` : correct answer is 90 days, not 60
4. `What is the dental insurance coverage?` : not in the document, so the model says it doesn't know

## Using your own document

Replace `sample.txt` with your own `.txt` file, or set `DOCUMENT_PATH` at the top of `rag.py` to a `.pdf` or `.txt` path. No other code changes are needed.

## Configuration

These values are at the top of `rag.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `DOCUMENT_PATH` | `sample.txt` | File to index |
| `CHROMA_DIR` | `./chroma_db` | Where the vector store is saved |
| `TOP_K` | `3` | Number of chunks retrieved per question |

Chunk size (400) and overlap (50) are set in `create_vector_store`. The models are set in `create_vector_store` (`nomic-embed-text`) and in the `ChatOllama(...)` call in `__main__` (`llama3.2`).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `python` is not recognized (Windows) | Reinstall Python with "Add python.exe to PATH" ticked, or use `py -3.13` |
| Script hangs or fails after "Created N chunks" | Ollama is not running, or a model is missing. Check `ollama list` |
| `model "nomic-embed-text" not found` | Run `ollama pull nomic-embed-text` |
| `ModuleNotFoundError` | Activate the virtual environment, then run `pip install -r requirements.txt` |
| `sample.txt not found` | Run the script from the folder that contains `sample.txt` |
| Duplicate chunks in results | Should not happen, since `chroma_db/` is deleted on each run. Make sure you are running the latest script |