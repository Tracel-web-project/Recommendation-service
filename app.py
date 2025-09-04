from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
from langchain_aws import ChatBedrock

app = Flask(__name__)
CORS(app)

# Initialize AWS Bedrock client
# Note: Ensure your environment has AWS credentials configured
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Initialize LangChain Bedrock wrapper (Claude v2 model)
# This assumes the user has access to the anthropic.claude-v2 model
llm = ChatBedrock(
    client=bedrock_client,
    model_id="anthropic.claude-v2"
)

# This single route now handles both health checks (GET) and recommendations (POST)
# The route is set to '/' to match the path NGINX forwards to the backend.
# The `methods` parameter is a list to accept both GET and POST.
@app.route('/api/recommendation', methods=['GET', 'POST'])
@app.route('/api/recommendation/', methods=['GET', 'POST'])
def handle_recommendation_requests():
    """
    Handles both health check and recommendation requests on the root path.
    """
    if request.method == 'GET':
        # Health check
        return "Recommendation Service is running", 200
    
    elif request.method == 'POST':
        # Recommendation request
        try:
            data = request.json
            user_query = data.get("query", "")

            if not user_query:
                return jsonify({"error": "Query is required"}), 400

            # Invoke Bedrock via LangChain
            response = llm.invoke(user_query)

            return jsonify({"recommendation": response.content})
        except Exception as e:
            # General error handling
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=False)
