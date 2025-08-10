from flask import Flask, request, jsonify
import google.generativeai as genai
from flask_cors import CORS 
import uuid
import os
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)  # It's recommended to specify domains in production, e.g., CORS(app, resources={r"/chat": {"origins": "https://your-frontend-domain.com"}})

# --- Configuration ---
# It's best practice to load the API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_api_key_here") 
MODEL_NAME = "gemini-1.5-flash" 
MAX_RESPONSE_TOKENS = 100 # This will be strictly enforced now

# The system prompt that defines the AI's persona and rules.
SYSTEM_PROMPT = """
You are a mature, non-judgmental friend/Psychologist for teenagers. Provide helpful, honest advice on any topic while maintaining appropriate boundaries. Strictly reply in 2 to 3 sentences or bullet points no more than that!.

Guidelines:
1. Always respond in a friendly, conversational tone using simple language
2. For sensitive topics:
   - Offer thoughtful guidance without graphic details
   - Suggest consulting trusted adults when appropriate
3. Structure responses clearly:
   • Use bullet points for multiple ideas
   • Add line breaks between concepts
4. Never say "I can't answer that" - instead:
   - Rephrase sensitive topics positively
   - Focus on general principles
   - Redirect to appropriate resources
"""

# --- Model Initialization ---
# Configure the Gemini client
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model with the system prompt. This prompt sets the base context.
model = genai.GenerativeModel(
    MODEL_NAME,
    system_instruction=SYSTEM_PROMPT,
    # Safety settings are adjusted to be less restrictive.
    safety_settings={
        'HATE': 'BLOCK_NONE',
        'HARASSMENT': 'BLOCK_NONE',
        'SEXUAL': 'BLOCK_NONE',
        'DANGEROUS': 'BLOCK_NONE'
    }
)

# --- In-Memory Session Storage ---
sessions = {}

# --- UPDATED ChatSession Class ---
# This class is now simplified to only manage history, not a stateful chat object.
class ChatSession:
    def __init__(self, session_id):
        self.id = session_id
        self.history = []  # Manually managed history
        self.created_at = datetime.now()
        self.last_used = datetime.now()
    
    def add_message(self, role, text):
        """Adds a message to the session's history in the correct format."""
        # The history format must be a list of dicts with 'role' and 'parts'.
        self.history.append({"role": role, "parts": [text]})
        self.last_used = datetime.now()
    
    def get_gemini_history(self):
        """Returns the history formatted for the generate_content API call."""
        return self.history

# --- Session Management ---
def get_session(session_id=None):
    """Retrieves an existing session or creates a new one."""
    if session_id and session_id in sessions:
        session = sessions[session_id]
        session.last_used = datetime.now()
        return session
    
    # Create a new session if no ID is provided or if the ID is not found
    new_id = session_id or f"session_{uuid.uuid4()}"
    session = ChatSession(new_id)
    sessions[new_id] = session
    return session

# --- Response Extraction ---
def extract_response_text(response):
    """Safely extracts text from the Gemini response, handling potential errors."""
    try:
        # The .text attribute is the most direct way to get the output.
        return response.text
    except ValueError:
        # Fallback for cases where the response structure is unexpected
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    return "".join(part.text for part in candidate.content.parts)
    except Exception:
        # Catch any other unforeseen errors.
        pass
    
    # Check for safety blocking, which is a common reason for no response
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        return f"Response blocked due to: {response.prompt_feedback.block_reason.name}"
        
    return "I'm sorry, I couldn't generate a response for that. Please try again."

# --- UPDATED Chat Endpoint ---
# This endpoint now uses the stateless `generate_content` method for reliability.
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
        
    query = data.get('query')
    session_id = data.get('session_id')
    
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400
    
    try:
        session = get_session(session_id)
        
        # Add the user's new message to our manually managed history.
        session.add_message("user", query)
        
        # Define the generation config for THIS specific call to enforce the token limit.
        generation_config = genai.GenerationConfig(
            max_output_tokens=MAX_RESPONSE_TOKENS,
            # You can also set other parameters like temperature here if needed
            # temperature=0.7 
        )

        # Send the entire conversation history to the model on each turn.
        # This stateless approach ensures rules are applied every time.
        response = model.generate_content(
            session.get_gemini_history(),
            generation_config=generation_config
        )
        
        answer = extract_response_text(response)
        
        # Add the model's response to our history to maintain the conversation context.
        session.add_message("model", answer)
        
        cleanup_old_sessions()
        
        return jsonify({
            "response": answer,
            "session_id": session.id,
            "model": MODEL_NAME,
            "history_length": len(session.history)
        })
    
    except Exception as e:
        # Log the full error for easier debugging
        print(f"An error occurred: {e}") 
        return jsonify({
            "error": "An internal error occurred. Please check the server logs.",
            "session_id": session_id
        }), 500

# --- Session Cleanup ---
def cleanup_old_sessions(max_age_seconds=3600, max_sessions=100):
    """Cleans up old or excess chat sessions to manage memory."""
    now = datetime.now()
    
    # Remove sessions that haven't been used for more than max_age_seconds
    expired_keys = [
        sid for sid, session in sessions.items()
        if (now - session.last_used).total_seconds() > max_age_seconds
    ]
    for key in expired_keys:
        sessions.pop(key, None)
    
    # If we still have too many sessions, remove the oldest ones
    if len(sessions) > max_sessions:
        # Sort sessions by last_used time, oldest first
        sorted_sessions = sorted(sessions.items(), key=lambda item: item[1].last_used)
        num_to_remove = len(sessions) - max_sessions
        for i in range(num_to_remove):
            key_to_remove = sorted_sessions[i][0]
            sessions.pop(key_to_remove, None)

# --- Other Endpoints ---
@app.route('/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Deletes a specific session."""
    if session_id in sessions:
        sessions.pop(session_id, None)
        return jsonify({"status": f"Session {session_id} deleted successfully"})
    return jsonify({"error": "Session not found"}), 404

@app.route('/model-info', methods=['GET'])
def model_info():
    """Provides information about the currently configured model."""
    return jsonify({
        "model": MODEL_NAME,
        "system_prompt": SYSTEM_PROMPT,
        "max_response_tokens": MAX_RESPONSE_TOKENS
    })

# --- Main Application Runner ---
if __name__ == '__main__':
    # Use debug=False in a production environment
    app.run(host='0.0.0.0', port=5000, debug=True)