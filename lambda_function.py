import boto3
import json

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

def lambda_handler(event, context):
    # CORS handling for API Gateway
    if event['httpMethod'] == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type, X-Amz-Date, Authorization, X-Api-Key',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps('Preflight OK')
        }

    try:
        # Parse the incoming JSON request body
        body = json.loads(event['body'])
        user_message = body.get('message', '')
        history = body.get('history', [])

        # Construct the conversation prompt
        conversation = ""
        for turn in history:
            conversation += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        conversation += f"User: {user_message}\nAssistant:"

        # Create payload for the model request
        request_body = {
            "inputText": conversation,
            "textGenerationConfig": {
                "maxTokenCount": 300,
                "temperature": 0.7,
                "topP": 0.9,
                "stopSequences": []
            }
        }

        # Invoke the Bedrock model
        response = bedrock.invoke_model(
            modelId='amazon.titan-text-express-v1',
            body=bytes(json.dumps(request_body), 'utf-8'),
            contentType='application/json',
            accept='application/json'
        )

        # Parse the response and return the generated reply
        result = json.loads(response['body'].read())
        reply = result.get('results', [{}])[0].get('outputText', '')

        # Return the API response with CORS headers
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type, X-Amz-Date, Authorization, X-Api-Key',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps({'response': reply})
        }

    except Exception as e:
        # Return an error response in case of issues
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type, X-Amz-Date, Authorization, X-Api-Key',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps({'error': str(e)})
        }
