# **LLM Chat Web UI with Ollama and Docker**

This project provides a simple web-based chat interface to interact with an LLM (in this instance the Qwen3 language model, specifically qwen3:4b), running locally using Ollama and managed with Docker Compose.

## **Features**

* **Local LLM:** Runs the Ollama server and Qwen3 model entirely on your local machine.  
* **Web Interface:** Provides a user-friendly chat UI accessible through your web browser.  
* **Dockerized:** Uses Docker Compose for easy setup, management, and dependency isolation.  
* **GPU Acceleration (Optional):** Can be configured to leverage your NVIDIA GPU for faster model inference.  
* **Persistent Model Storage:** Downloads the LLM model once and stores it persistently using a Docker volume.  
* **Automatic Model Download:** The web server checks if the required model is available and downloads it automatically on first run if needed.
* **Persistent conversation history:** Persists conversation history in the backend, sending full history to the model for continued chat context.

## **Technology Stack**

* [**Ollama**](https://ollama.com/)**:** Runs the Qwen3 large language model locally and provides an OpenAI-compatible API.  
* [**Docker**](https://www.docker.com/) **& [Docker Compose](https://docs.docker.com/compose/):** Containerizes the application and manages the services.  
* [**Python**](https://www.python.org/)**:**  
  * [**Flask**](https://flask.palletsprojects.com/)**:** Micro web framework used for the backend server and API.  
  * [**OpenAI Python Client**](https://github.com/openai/openai-python)**:** Used to interact with Ollama's OpenAI-compatible chat API endpoint.  
  * [**Requests**](https://requests.readthedocs.io/)**:** Used to interact with Ollama's specific API for pulling models.  
* **HTML, CSS (Tailwind), JavaScript:** For the frontend chat interface.

## **Project Structure**

qwen_chat/  
├── docker-compose.yml      # Defines the ollama and web_server services  
├── README.md               # This file  
│  
└── web_server/  
   ├── app.py              # Flask backend server code  
   ├── Dockerfile          # Instructions to build the web_server image  
   ├── requirements.txt    # Python dependencies  
   ├── .dockerignore       # Files/folders to ignore during Docker build  
   │  
   └── templates/  
       └── index.html      # HTML/CSS/JS for the chat interface

## **Prerequisites**

1. **Docker Desktop:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) for your operating system (Windows, macOS, or Linux).  
2. **(Optional - for GPU Acceleration on NVIDIA):**  
   * **NVIDIA Drivers:** Ensure you have the latest drivers for your NVIDIA GPU installed.  
   * **For Windows:** Docker Desktop uses WSL 2\. Ensure your Windows, NVIDIA drivers, and Docker Desktop are up-to-date. GPU support should work automatically if prerequisites are met. See [Docker Docs for GPU on Windows](https://docs.docker.com/desktop/features/gpu/).  
   * **For Linux:** Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). After installation, configure the Docker daemon:  
     
	 > sudo nvidia-ctk runtime configure --runtime=docker
	 
	 > sudo systemctl restart docker

## **Setup and Running**

1. **Clone or Download:** Get the project files onto your local machine.  
2. **Navigate:** Open a terminal or PowerShell window and navigate into the main project directory (qwen_chat/).  
3. **(Optional - Enable GPU):**  
   * Edit the docker-compose.yml file.  
   * Find the ollama service definition.  
   * Uncomment the deploy: section to enable GPU access.  
4. **Build and Start:** Run the following command:  
   
   > docker-compose up --build -d

   * --build: Builds the web_server image (only needed the first time or after code changes).  
   * -d: Runs the containers in detached mode (in the background).  
5. **Wait for Initialization:** The first time you run this, Ollama needs to download the qwen3:4b model (several gigabytes). The web_server will wait for Ollama to start and automatically trigger the download if necessary. You can monitor the progress by checking the logs:  
   
   Check Ollama server logs (shows model download progress)  
   
   > docker-compose logs -f ollama

   Check Web server logs (shows connection attempts and server start)  
   
   > docker-compose logs -f web_server

   Wait until the logs indicate the model is downloaded and the Flask server has started.

6. You may use any model available to Ollama by updating the following line in *app.py*. Make your change and follow Step 4 again.

	> MODEL_NAME = "qwen3:4b"

## **Usage**

1. Access the Web UI: Open your web browser and go to:  
   http://localhost:5001  
2. **Chat:** Enter your prompts in the input box and press Enter. The response from the Qwen3 model will appear.  
3. **Stopping:** When you are finished, run the following command in the terminal from the project directory:  
   
   > docker-compose down

   *(Optional: To remove the downloaded model data as well, use docker-compose down -v)*

## **Restarting After Closing Docker Desktop**

If you shut down Docker Desktop and want to use the chat again later:

1. **Start Docker Desktop.**  
2. **Navigate:** Open a terminal/PowerShell in the project directory.  
3. **Start Containers:** Run:  
   
   > docker-compose up -d

   *(No --build is needed unless you changed the code)*.  
   
4. **Wait:** Give the Ollama server container ~15-30 seconds to initialize.  
5. **Access:** Open http://localhost:5001 in your browser.

## **How it Works**

* **docker-compose.yml:** Orchestrates the two services.  
  * **ollama service:** Runs the official ollama/ollama image. It exposes port 11434 internally for the web server to connect to. It uses a named volume (ollama_data) to persistently store downloaded models in /root/.ollama inside the container. The optional deploy key configures NVIDIA GPU access.  
  * **web_server service:** Builds a custom image based on the web_server/Dockerfile. It installs Python, Flask, and other dependencies. It runs app.py, which starts a Flask web server on port 5001. Port 5001 is mapped to localhost:5001 on your host machine. It connects to the ollama service using its service name (ollama_server) thanks to Docker Compose's networking.  
  * **qwen_net network:** A custom bridge network allowing the web_server and ollama services to communicate via their service names.  
* **app.py:**  
  * Waits for the Ollama server to be accessible.  
  * Checks if the qwen3:4b model exists using Ollama's API.  
  * If the model doesn't exist, it uses the requests library to call Ollama's /api/pull endpoint to download it, streaming the progress to the logs.  
  * Starts the Flask web server.  
  * Provides a route to serve the index.html page.  
  * Provides a chat API endpoint that receives user messages (POST request), sends them to the Ollama /v1/chat/completions endpoint using the OpenAI client library, and returns the model's response.  
* **index.html:** A simple HTML page using Tailwind CSS for styling and basic JavaScript to handle form submission (sending the prompt to the /chat endpoint via fetch) and display the chat history.

## **Troubleshooting**

* **Error: model 'qwen3:4b' not found (in web UI or logs):** This usually means the model download was interrupted or the volume was cleared. The app.py script should handle this on startup, but ensure the ollama_data volume exists (docker volume ls). Running docker-compose up --build -d should trigger the download check again.  
* **Error: Connection refused or Name or service not known (in web\_server logs):** Make sure both services are running (docker ps) and are correctly configured to use the qwen_net in docker-compose.yml. Try restarting with docker-compose down && docker-compose up -d.  
* **Web UI shows Error: HTTP error! Status: 500:** Check the web_server logs (docker-compose logs -f web_server) for the detailed Python error traceback.  
* **Slow responses / High CPU usage:** This likely means Ollama is not using the GPU.  
  * **Verify GPU Configuration:** Ensure the deploy: section in docker-compose.yml is uncommented *and* that the NVIDIA Container Toolkit (Linux) or WSL 2 GPU Passthrough (Windows) is correctly set up on your host machine.  
  * **Check Ollama Logs:** Look at the very beginning of the ollama service logs after starting it: docker-compose logs ollama. You should see messages indicating it detected the NVIDIA GPU. If it mentions CPU or ROCm (for AMD), the NVIDIA GPU is not being used.  
  * **Check GPU Usage on Host:** Run nvidia-smi on your host machine *while* sending a prompt. You should see ollama or a related process utilizing GPU memory and compute.