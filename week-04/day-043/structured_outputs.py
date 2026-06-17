import json
import os
from typing import Literal

import anthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError


class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=10)


RoleType = Literal["assistant", "user"]

SYSTEM_PROMPT: str = """You are a sentiment analyser.

Always return your answers only in JSON format. No prose, no comments. 
Do not add any extra strings other than JSON format.
Your responses must match exactly the schema below:

{
    "sentiment": "positive" | "negative" | "neutral",
    "confidence": float between 0.0 and 1.0,
    "reasoning": only one sentence string
}

If the text contains both positive and negative elements with no clearly dominant 
sentiment, classify it as "neutral".
"""

load_dotenv()
client = anthropic.Anthropic()


def build_message(role: RoleType, input_text: str) -> MessageParam:
    """
    Builds a MessageParam object with the given input according to the role selection

    Arguments:
        role {RoleType} -- The role selection
        input_text {str} -- The input text

    Returns:
        MessageParam -- A MessageParam object
    """
    return {
        "role": role,
        "content": input_text
    }

def append_message(history: list[MessageParam], role: RoleType, input_text: str) -> None:
    """
    Appends a message to the history of the messages

    Arguments:
        history {list[MessageParam]} -- The history of the messages
        role {RoleType} -- The role selection
        input_text {str} -- The input text
    """
    message_to_append: MessageParam = build_message(role, input_text)
    history.append(message_to_append)

def analyse_sentiment(text: str) -> tuple[SentimentAnalysis | None, str | None]:
    """
    Analyzes the sentiment of the response for the input text

    Arguments:
    text {str} -- The input text

    Returns:
    SentimentAnalysis -- The sentiment of the input text
    """
    message: list[MessageParam] = []

    append_message(history=message, role="user", input_text=text)
    response = client.messages.create(
        model=os.environ["ANTHROPIC_MODEL_NAME"],
        max_tokens=512,
        temperature=0.0,
        system=SYSTEM_PROMPT,
        messages=message
    )

    response_text: str = response.content[0].text
    cleared_text: str = response_text.strip()
    if cleared_text.startswith("```"):
        cleared_text = cleared_text.split("```")[1]
        if cleared_text.startswith("json"):
            cleared_text = cleared_text.replace("json", "", 1)

    try:
        parsed_text = json.loads(cleared_text)
        result_sentiment = SentimentAnalysis.model_validate(parsed_text)
    except json.decoder.JSONDecodeError:
        print(f"Error decoding cleared text: {cleared_text}")
        return None, "json_parse_error"
    except ValidationError as e:
        print(f"Validation failed for: {cleared_text}. Error: {e}")
        return None, "validation_error"

    return result_sentiment, None

if __name__ == "__main__":
    inputs = [
        "This product completely changed my morning routine. I love it.",
        "Arrived broken. Terrible packaging. Never buying again.",
        "It's okay I guess. Does what it says.",
    ]
    for inp in inputs:
        result = analyse_sentiment(inp)
        print(f"""
Input: {inp}""")
        if result[0]:
            print(f"""Result: Sentiment: {result[0].sentiment}
        Confidence: {result[0].confidence}
        Reasoning: {result[0].reasoning}""")
        else:
            print(f"Result: {result[1]}")