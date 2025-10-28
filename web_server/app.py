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
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Initialize the OpenAI client to connect to Ollama's V1 API
client = OpenAI(
    base_url=f"{OLLAMA_HOST}/v1",
    api_key='ollama',  # required, but a dummy key for Ollama
)

MODEL_NAME = "qwen3:4b"

# --- Global variable to store conversation history ---
# We can optionally start with a system prompt
conversation_history = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
# If you don't want a system prompt, initialize like this:
# conversation_history = []
# ----------------------------------------------------

def wait_for_ollama(timeout=600):
    """
    Waits for the Ollama server to be ready and ensures the model is pulled.
    """
    print("Connecting to Ollama server...")
    start_time = time.time()

    # 1. Wait for the server to be alive
    while time.time() - start_time < timeout:
        try:
            response = requests.get(OLLAMA_HOST, timeout=5) # Add timeout
            response.raise_for_status()
            print("Ollama server is alive (HTTP check successful).")
            break
        except requests.exceptions.ConnectionError:
            print("Ollama not ready (connection refused)... retrying in 5 seconds.")
            time.sleep(5)
        except requests.exceptions.Timeout:
             print("Ollama not ready (HTTP check timed out)... retrying in 5 seconds.")
             time.sleep(5)
        except requests.exceptions.RequestException as e:
             print(f"Ollama not ready (HTTP error {e})... retrying in 5 seconds.")
             time.sleep(5)
        except Exception as e:
            print(f"An unexpected error occurred while waiting for Ollama HTTP check: {e}")
            traceback.print_exc()
            time.sleep(5)
    else:
        print("Ollama server connection timed out.")
        return False

    # 2. Check if the model is already pulled using the OpenAI client
    try:
        model_list_response = None
        models_data = None
        model_exists = False

        try:
            print("Attempting to list models from Ollama via OpenAI client...")
            model_list_response = client.models.list()
            print(f"Received model list response object: {type(model_list_response)}")

            if model_list_response:
                # Use getattr for safer access in case the structure is unexpected
                models_data = getattr(model_list_response, 'data', None)
                print(f"Extracted models_data: {type(models_data)}")
            else:
                print("client.models.list() returned None or empty response.")

        except APIConnectionError as conn_err:
             print(f"Connection error during client.models.list(): {conn_err}")
             # Let's try pulling if we can't list
             model_exists = False
             print("Proceeding to check pull necessity due to connection error during list.")
        except Exception as list_error:
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
                 # Check the 'id' or 'name' attribute - Ollama might use 'name' in some contexts
                 model_exists = any(
                    getattr(model, 'id', getattr(model, 'name', None)) == MODEL_NAME
                    for model in models_data
                 )
                 print(f"Model exists check result: {model_exists}")
            else:
                 print(f"Models data received ({type(models_data)}), but it's not a list as expected.")
                 model_exists = False # Treat non-list as model not found
        elif not model_exists and model_list_response is not None:
             # This case handles when listing succeeded but models_data extraction failed
             print("Model listing seemed successful but failed to extract model data correctly.")
             model_exists = False


        # --- Proceed based on model_exists status ---
        if model_exists:
            print(f"Model '{MODEL_NAME}' is already available.")
            return True

        # 3. If model is not pulled, pull it using the /api/pull endpoint
        print(f"Model '{MODEL_NAME}' not found or listing failed/ambiguous. Pulling model...")

        pull_url = f"{OLLAMA_HOST}/api/pull"
        pull_data = {"name": MODEL_NAME, "stream": True} # Stream True gives progress

        print(f"Sending POST request to {pull_url}...")
        try:
            with requests.post(pull_url, json=pull_data, stream=True, timeout=timeout) as response:
                print(f"Received response status code: {response.status_code}")
                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

                last_status = ""
                print("Streaming pull response:")
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if 'status' in chunk:
                                status = chunk['status']
                                if status != last_status:
                                    # Print progress updates concisely
                                    print(f"  ... {status}", end='\r')
                                    last_status = status
                            if 'error' in chunk:
                                print(f"\nError message during pull: {chunk['error']}") # Newline before error
                                return False
                        except json.JSONDecodeError:
                            print(f"\n  Could not decode JSON line: {line}")
                        except Exception as chunk_e:
                             print(f"\n  Error processing chunk: {chunk_e}")
                print("\nModel pull stream finished.") # Newline after progress ends

        except requests.exceptions.Timeout:
            print(f"\nError: Timeout occurred while pulling model '{MODEL_NAME}'.")
            return False
        except requests.exceptions.RequestException as req_err:
             print(f"\nError: Request failed during model pull: {req_err}")
             return False
        except Exception as pull_err:
             print(f"\nAn unexpected error occurred during model pull: {pull_err}")
             traceback.print_exc()
             return False


        # Verify model exists *after* pulling using the client again
        try:
            print("Verifying model presence after pull attempt...")
            final_models_list = client.models.list().data
            if isinstance(final_models_list, list) and any(
                getattr(m, 'id', getattr(m, 'name', None)) == MODEL_NAME for m in final_models_list):
                 print(f"Model '{MODEL_NAME}' successfully verified after pull.")
                 return True
            else:
                 print(f"Model '{MODEL_NAME}' still not found after pull attempt. Final list: {final_models_list}")
                 return False
        except Exception as verify_e:
             print(f"Error verifying model after pull: {verify_e}")
             return False # Assume failure if verification fails


    except Exception as e:
        # Catch any other unexpected errors in the main try block
        print(f"An unexpected error occurred in wait_for_ollama: {e}")
        print("Detailed traceback:")
        traceback.print_exc()
        return False

# --- Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles the chat API request, maintaining history."""
    global conversation_history # Use the global history list
    try:
        user_message = request.json['message']
        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        print(f"Received message: {user_message}")

        # --- Append user message to history ---
        user_message_dict = {'role': 'user', 'content': user_message}
        conversation_history.append(user_message_dict)
        # -------------------------------------

        # --- Send the *entire* history to Ollama ---
        print(f"Sending history (length {len(conversation_history)}) to model...")
        chat_completion = client.chat.completions.create(
            messages=conversation_history, # Send the whole list
            model=MODEL_NAME,
            stream=False, # Keeping stream False for simplicity
        )
        # -------------------------------------------

        bot_response = chat_completion.choices[0].message.content
        print(f"Sending response: {bot_response[:50]}...")

        # --- Append bot response to history ---
        bot_response_dict = {'role': 'assistant', 'content': bot_response}
        conversation_history.append(bot_response_dict)
        # ------------------------------------

        return jsonify({"response": bot_response}) # Return only the latest response

    except Exception as e:
        print(f"Error during chat: {e}")
        traceback.print_exc() # Print full traceback for server logs
        # Try to remove the last user message from history if an error occurred before getting a response
        if conversation_history and conversation_history[-1]['role'] == 'user':
            conversation_history.pop()
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route('/reset', methods=['POST'])
def reset_chat():
    """Clears the conversation history."""
    global conversation_history
    # Reset to initial state (e.g., just the system prompt if used)
    conversation_history = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    # If you didn't use a system prompt, reset like this:
    # conversation_history = []
    print("Chat history reset.")
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    # Wait for Ollama to be ready before starting the web server
    if not wait_for_ollama():
        print("Failed to connect to Ollama or pull model. Exiting.")
        exit(1)

    # Start the Flask server
    print(f"Starting Flask server on http://0.0.0.0:5001")
    # Make sure debug=True is set for development
    app.run(host='0.0.0.0', port=5001, debug=True)

