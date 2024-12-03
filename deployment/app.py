from flask import Flask, request, jsonify, render_template
import logging
from deployment.exception import CustomException
from deployment.logger import logging as app_logger
from deployment.prompt_template import PromptTemplate  

# Initialize Flask app
app = Flask(__name__)

# Configure logging
app_logger.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = app_logger.getLogger()

# Initialize PromptTemplate
prompt_template = PromptTemplate()

# Route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle POST request for questions
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        if 'question' not in data:
            logger.error("Question key not found in the request data.")
            return jsonify({"error": "Missing 'question' in request"}), 400

        # Extract user question
        user_question = data['question']
        if not user_question.strip():
            logger.error("Received empty question.")
            return jsonify({"error": "Question cannot be empty"}), 400

        # Generate the response using the PromptTemplate
        logger.info(f"Received question: {user_question}")
        response = prompt_template.generate(user_question)

        # Return the response
        return jsonify({"answer": response}), 200

    except CustomException as e:
        logger.error(f"CustomException occurred: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500


# Start Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
