import boto3
import json
import streamlit as st

session = boto3.Session(
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name=st.secrets["AWS_REGION"]
)

s3 = session.client("s3")

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def get_bedrock_explanation(label):
    prompt = f"""
    You are a car expert.

    Your task is to:
    1. Start your response with exactly this sentence: "It seems like you're looking for **{label}**!"
    2. After that, write exactly two sentences describing the general location and function of the car part '{label}'.
    3. Do not include any extra information or disclaimers.

    Only follow the instructions above and write exactly 3 sentences in total.
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
