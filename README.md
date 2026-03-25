# Local Ollama Chat Console

A web console for interacting with local Ollama LLMs. This project is a local web based chat application with persistent session management, and integration with Ollama's API for real-time metrics and VRAM control.

## Features

-   **Persistent Multi-Session Sidebar:** Chat threads are stored locally in the browser (`localStorage`), allowing for persistent history without needing a backend database.
-   **Advanced Model Configuration:** Granular control over `Temperature`, `Top_P`, `Top_K`, `Num_Ctx`, `Repeat_Penalty`, and `Seed`.
-   **Real-time Performance Metrics:** Live display of Input/Output tokens, Tokens Per Second (TPS), and total duration for every response.
-   **Markdown & Syntax Highlighting:** Integrated support for rich text and code blocks using `marked.js` and `highlight.js`.
-   **VRAM Management:** Automatic unloading of models from VRAM when switching models or shutting down the server to free up system resources.
-   **VRAM Estimation:** Real-time calculation of VRAM requirements (Weights + KV Cache + Overhead) based on model architecture and quantization levels.
-   **Export Chats:** Export chats to Markdown files.

## Architecture

The application follows a secure **Proxy Pattern**:
-   **Backend (FastAPI):** Acts as an asynchronous bridge between the browser and the Ollama API, handling SSE (Server-Sent Events) streaming.
-   **Frontend (JS/CSS/HTML):** A self-contained UI that manages state, parameters, and history entirely on the client side.

## Getting Started

### Prerequisites

-   **Python 3.8+**
-   **Ollama:** Must be installed and running locally. [Download Ollama here](https://ollama.com/).

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/localchatbot.git
cd localchatbot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```

### 4. Access the Console
Open your browser and navigate to `http://localhost:8000`

## Configuration

The console allows you to tune Ollama parameters on the fly via the **Config** panel:
-   **System Instructions:** Define the persona and constraints for the AI.
-   **Num_Ctx:** Adjust the context window size (automatically updates VRAM estimates).
- 	**Ollama Parameters:** Adjust `Temperature`, `Top_P`, `Top_K`, `Num_Ctx`, `Repeat_Penalty`, and `Seed`
-   **Unload Model:** Models are sent with `keep_alive: 0` during switches to ensure immediate VRAM release.

