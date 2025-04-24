import boto3
import json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def get_bedrock_explanation(label):
    prompt = f"""This '{label}' is a part name of a car.
                Start with the sentence saying 'It seems like you're looking for '**{label}**''!
                Then describe the location and role of the car part named '{label}' in two sentences.
             """

    body = {
        "inputText": prompt,
        "textGenerationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "maxTokenCount": 300,
            "stopSequences": []
        }
    }

    response = bedrock.invoke_model(
        modelId="amazon.titan-text-lite-v1",
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response['body'].read())
    return result.get("results", [{}])[0].get("outputText", "[No output]")
