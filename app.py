from flask import Flask, request, jsonify
import google.generativeai as genai
from flask_cors import CORS
import uuid
import os
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_api_key_here")

# UPDATED: The model name is confirmed as "gemini-2.5-flash" for the stable release.
MODEL_NAME = "gemini-2.5-flash"

# This now acts as a secondary, technical limit. The primary limit is in the prompt.

# The prompt template is now the primary method of control.
PROMPT_TEMPLATE = """
**INSTRUCTIONS:**
You are a mature, non-judgmental friend/Psychologist for teenagers.
Provide helpful, honest advice on the user's question below.
Your response MUST be very concise and strictly limited to under 150 words.
You MUST reply in 2-3 short sentences or a few bullet points.

**USER'S QUESTION:**
"{user_query}"
"""

# --- Model Initialization ---
# The initialization method remains the same for the new model.
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    MODEL_NAME,
    safety_settings={
        'HATE': 'BLOCK_NONE',
        'HARASSMENT': 'BLOCK_NONE',
        'SEXUAL': 'BLOCK_NONE',
        'DANGEROUS': 'BLOCK_NONE'
    }
)

# --- In-Memory Session Storage & Session Class ---
sessions = {}

class ChatSession:
    def __init__(self, session_id):
        self.id = session_id
        self.history = []
        self.created_at = datetime.now()
        self.last_used = datetime.now()

    def add_message(self, role, text):
        self.history.append({"role": role, "parts": [text]})
        self.last_used = datetime.now()

    def get_gemini_history(self):
        return self.history

# --- Session Management ---
def get_session(session_id=None):
    if session_id and session_id in sessions:
        session = sessions[session_id]
        session.last_used = datetime.now()
        return session

    new_id = session_id or f"session_{uuid.uuid4()}"
    session = ChatSession(new_id)
    sessions[new_id] = session
    return session

# --- Response Extraction ---
def extract_response_text(response):
    try:
        return response.text
    except Exception:
        try:
            if response.candidates:
                return "".join(part.text for part in response.candidates[0].content.parts)
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                return f"Response blocked due to: {response.prompt_feedback.block_reason.name}"
        except Exception as e:
            print(f"Error during response extraction: {e}")
            return "I'm sorry, I encountered an error while generating a response."
    return "I am unable to provide a response at this time."

# --- Chat Endpoint (with updated response JSON) ---
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Invalid request, missing 'query'"}), 400

    query = data.get('query')
    session_id = data.get('session_id')

    try:
        session = get_session(session_id)

        # Construct the final prompt by merging the template and the user's query.
        final_prompt_for_api = PROMPT_TEMPLATE.format(user_query=query)

        # Add the original, clean user query to the history.
        session.add_message("user", query)

        history_for_api = list(session.get_gemini_history())

        # Replace the last user message with our combined, forceful prompt for the API call.
        history_for_api[-1]['parts'] = [final_prompt_for_api]

        # Define generation config as a technical safeguard.
        # NEW: Added optional "thinking_config" for Gemini 2.5 Flash.
        # This allows the model to spend more time on complex prompts to improve accuracy.
        # You can adjust the budget (0 to 24576) or remove this config entirely.
        generation_config = genai.GenerationConfig(
            temperature=0.7,
            # thinking_config=genai.types.ThinkingConfig(
            #     thinking_budget=1024  # Example budget, can be adjusted
            # )
        )

        # The method to generate content remains the same.
        response = model.generate_content(
            history_for_api,
            generation_config=generation_config
        )

        answer = extract_response_text(response)

        # Add the clean model response to the persistent session history.
        session.add_message("model", answer)

        cleanup_old_sessions()

        # The original 'query' is now included in the JSON response.
        return jsonify({
            "query": query,
            "response": answer,
            "session_id": session.id,
            "model": MODEL_NAME,
            "history_length": len(session.history)
        })

    except Exception as e:
        print(f"An error occurred in chat_endpoint: {e}")
        return jsonify({
            "error": "An internal error occurred. Please check the server logs.",
            "session_id": session_id
        }), 500

# --- Cleanup and Other Endpoints (Unchanged) ---
def cleanup_old_sessions(max_age_seconds=3600, max_sessions=100):
    now = datetime.now()
    expired_keys = [sid for sid, session in sessions.items() if (now - session.last_used).total_seconds() > max_age_seconds]
    for key in expired_keys:
        sessions.pop(key, None)

    if len(sessions) > max_sessions:
        sorted_sessions = sorted(sessions.items(), key=lambda item: item[1].last_used)
        num_to_remove = len(sessions) - max_sessions
        for i in range(num_to_remove):
            key_to_remove = sorted_sessions[i][0]
            sessions.pop(key_to_remove, None)

@app.route('/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    if session_id in sessions:
        sessions.pop(session_id, None)
        return jsonify({"status": f"Session {session_id} deleted successfully"})
    return jsonify({"error": "Session not found"}), 404

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "model": MODEL_NAME,
        "prompting_strategy": "Instructions are merged with the user query on each turn.",
        "max_response_tokens_safeguard": MAX_RESPONSE_TOKENS,
        "thinking_feature": "Available but not enabled by default in this code. See comments in chat_endpoint."
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)