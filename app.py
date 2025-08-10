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
MODEL_NAME = "gemini-1.5-flash" 
MAX_RESPONSE_TOKENS = 80 # Lowered slightly to be even more strict

# The system prompt that defines the AI's persona and rules.
SYSTEM_PROMPT = """
You are a mature, non-judgmental friend/Psychologist for teenagers. Provide helpful, honest advice on any topic while maintaining appropriate boundaries. Strictly reply in 2 to 3 sentences or bullet points no more than that!.
"""

# --- Model Initialization ---
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    MODEL_NAME,
    system_instruction=SYSTEM_PROMPT,
    safety_settings={
        'HATE': 'BLOCK_NONE',
        'HARASSMENT': 'BLOCK_NONE',
        'SEXUAL': 'BLOCK_NONE',
        'DANGEROUS': 'BLOCK_NONE'
    }
)

# --- In-Memory Session Storage ---
sessions = {}

# --- ChatSession Class ---
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
        # Fallback for safety or other issues
        try:
            if response.candidates:
                return "".join(part.text for part in response.candidates[0].content.parts)
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                return f"Response blocked due to: {response.prompt_feedback.block_reason.name}"
        except Exception as e:
            print(f"Error during response extraction: {e}")
            return "I'm sorry, I encountered an error while generating a response."
    return "I am unable to provide a response at this time."


# --- DEEPLY REVISED Chat Endpoint ---
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Invalid request, missing 'query'"}), 400
        
    query = data.get('query')
    session_id = data.get('session_id')
    
    try:
        session = get_session(session_id)
        
        # 1. Add the user's clean message to the session history
        session.add_message("user", query)
        
        # 2. Prepare the payload for the API call
        # We create a temporary copy of the history to modify.
        history_for_api = list(session.get_gemini_history())

        # 3. REINFORCE THE PROMPT
        # We add a new, forceful instruction at the end of the history.
        # This makes it the last thing the model reads.
        forceful_instruction = {
            "role": "user",
            "parts": [
                f"Remember your instructions. Be extremely brief. "
                f"STRICTLY 2-3 sentences or a few bullet points. "
                f"Your response must be short."
            ]
        }
        # We insert this instruction right before the user's actual last message.
        # This frames the user's query with our rule.
        history_for_api.insert(-1, forceful_instruction)


        # 4. DEFINE A STRICTER GENERATION CONFIG
        # We are adding a "stop" sequence. The model will stop generating if it
        # tries to write a double newline, which typically separates paragraphs.
        generation_config = genai.GenerationConfig(
            max_output_tokens=MAX_RESPONSE_TOKENS,
            stop_sequences=["\n\n"], # Stop if it tries to make a new paragraph
            temperature=0.7 # A slightly lower temperature can reduce verbosity
        )

        # 5. GENERATE CONTENT with the reinforced history and strict config
        response = model.generate_content(
            history_for_api,
            generation_config=generation_config
        )
        
        answer = extract_response_text(response)
        
        # 6. Add only the clean model response to the persistent history
        session.add_message("model", answer)
        
        cleanup_old_sessions()
        
        return jsonify({
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


# (The rest of the code for cleanup, delete, and info endpoints remains the same)
# --- Session Cleanup ---
def cleanup_old_sessions(max_age_seconds=3600, max_sessions=100):
    now = datetime.now()
    expired_keys = [
        sid for sid, session in sessions.items()
        if (now - session.last_used).total_seconds() > max_age_seconds
    ]
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
        "system_prompt": SYSTEM_PROMPT,
        "max_response_tokens": MAX_RESPONSE_TOKENS
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)