import os
import time
import requests
import json
import traceback # Import traceback for detailed error printing
from flask import Flask, render_template, request, jsonify
from openai import OpenAI, APIConnectionError

# Initialize the Flask app
app = Flask(__name__)

# Get the Ollama host from the environment variable
# This will be 'http://ollama_server:11434' inside the Docker network
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Initialize the OpenAI client to connect to Ollama's V1 API
client = OpenAI(
    base_url=f"{OLLAMA_HOST}/v1",
    api_key='ollama',  # required, but a dummy key for Ollama
)

MODEL_NAME = "qwen3:4b"

def wait_for_ollama(timeout=600): # Increased timeout for model download
    """
    Waits for the Ollama server to be ready and ensures the model is pulled.
    """
    print("Connecting to Ollama server...")
    start_time = time.time()

    # 1. Wait for the server to be alive
    while time.time() - start_time < timeout:
        try:
            # Use a more specific check that should return quickly
            response = requests.get(OLLAMA_HOST)
            response.raise_for_status()
            print("Ollama server is alive (HTTP check successful).")
            break
        except requests.exceptions.ConnectionError:
            print("Ollama not ready (connection refused)... retrying in 5 seconds.")
            time.sleep(5)
        except requests.exceptions.RequestException as e:
             print(f"Ollama not ready (HTTP error {e})... retrying in 5 seconds.")
             time.sleep(5)
        except Exception as e:
            print(f"An unexpected error occurred while waiting for Ollama HTTP check: {e}")
            time.sleep(5)
    else:
        print("Ollama server connection timed out.")
        return False

    # 2. Check if the model is already pulled using the OpenAI client
    try:
        model_list_response = None # Initialize
        models_data = None
        model_exists = False

        try:
            print("Attempting to list models from Ollama via OpenAI client...")
            model_list_response = client.models.list()
            print(f"Received model list response object: {type(model_list_response)}")
            # Check response structure carefully
            if model_list_response:
                models_data = getattr(model_list_response, 'data', None)
                print(f"Extracted models_data: {type(models_data)}")
            else:
                print("client.models.list() returned None or empty response.")

        except APIConnectionError:
             print("Connection error during client.models.list() - server might be temporarily busy.")
             # Assume model needs pulling if we can't list
             model_exists = False
        except Exception as list_error:
            # Catch potential errors *during* the list call, including potential NoneType iterations internally
            print(f"Error occurred specifically during client.models.list(): {list_error}")
            print("Detailed traceback for list error:")
            traceback.print_exc()
            # Assume model needs pulling if listing fails unexpectedly
            model_exists = False
            print("Proceeding to check pull necessity despite list error.")

        # Check model existence ONLY if listing was successful and returned data
        if models_data is not None:
            if isinstance(models_data, list):
                 print(f"Models data is a list (length {len(models_data)}). Checking for {MODEL_NAME}...")
                 # Use getattr for safer access to model.id
                 model_exists = any(getattr(model, 'id', None) == MODEL_NAME for model in models_data)
                 print(f"Model exists check result: {model_exists}")
            else:
                 print("Models data received, but it's not a list.")
                 model_exists = False # Treat non-list as model not found

        # --- Proceed based on model_exists status ---

        if model_exists:
            print(f"Model '{MODEL_NAME}' is already available.")
            return True

        # 3. If model is not pulled, pull it using the /api/pull endpoint
        print(f"Model '{MODEL_NAME}' not found or listing failed/ambiguous. Pulling model...")

        pull_url = f"{OLLAMA_HOST}/api/pull"
        pull_data = {"name": MODEL_NAME, "stream": True}

        print(f"Sending POST request to {pull_url}...")
        with requests.post(pull_url, json=pull_data, stream=True, timeout=timeout) as response:
            print(f"Received response status code: {response.status_code}")
            response.raise_for_status() # Raise an exception for bad status codes

            last_status = ""
            print("Streaming pull response:")
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        # print(f"  Chunk: {chunk}") # Uncomment for very verbose debugging
                        if 'status' in chunk:
                            status = chunk['status']
                            if status != last_status:
                                print(f"  ... {status}")
                                last_status = status
                        if 'error' in chunk:
                            print(f"Error message during pull: {chunk['error']}")
                            return False
                    except json.JSONDecodeError:
                        print(f"  Could not decode JSON line: {line}")
                    except Exception as chunk_e:
                         print(f"  Error processing chunk: {chunk_e}")


        print(f"Model '{MODEL_NAME}' pull stream finished.")
        # Verify model exists *after* pulling
        try:
            print("Verifying model presence after pull...")
            final_models_list = client.models.list().data
            if isinstance(final_models_list, list) and any(getattr(m, 'id', None) == MODEL_NAME for m in final_models_list):
                 print(f"Model '{MODEL_NAME}' successfully verified after pull.")
                 return True
            else:
                 print(f"Model '{MODEL_NAME}' still not found after pull attempt.")
                 return False
        except Exception as verify_e:
             print(f"Error verifying model after pull: {verify_e}")
             return False # Assume failure if verification fails


    except Exception as e:
        # This will catch errors during the pull process OR unexpected errors during checking
        print(f"An error occurred in the broader check/pull block: {e}")
        print("Detailed traceback for broader error:")
        traceback.print_exc() # Print full traceback for debugging
        return False

# --- Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles the chat API request."""
    try:
        user_message = request.json['message']
        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        print(f"Received message: {user_message}")

        chat_completion = client.chat.completions.create(
            messages=[{'role': 'user', 'content': user_message}],
            model=MODEL_NAME,
            stream=False,
        )

        bot_response = chat_completion.choices[0].message.content
        print(f"Sending response: {bot_response[:50]}...")
        return jsonify({"response": bot_response})

    except Exception as e:
        print(f"Error during chat: {e}")
        # Send the specific error message to the client for debugging
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Wait for Ollama to be ready before starting the web server
    if not wait_for_ollama():
        print("Failed to connect to Ollama or pull model. Exiting.")
        exit(1)

    # Start the Flask server
    print(f"Starting Flask server on http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)

