import os
import json
import time
import requests
import openai
import copy

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")

API_KEY_2 = os.getenv("API_KEY_2")
API_BASE_2 = os.getenv("API_BASE_2")

MAX_TOKENS = os.getenv("MAX_TOKENS")
TEMPERATURE = os.getenv("TEMPERATURE")

DEBUG = int(os.environ.get("DEBUG", "0"))


def generate_together(
    model,
    messages,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    api_base=API_BASE,
    api_key=API_KEY,
    streaming=False,
):

    logger.info(
        f"Input data: model={model}, messages={messages}, max_tokens={max_tokens}, temperature={temperature}"
    )

    output = None

    for sleep_time in [1, 2, 4, 8, 16, 32]:

        try:

            endpoint = f"{api_base}/chat/completions"

            logger.info(f"Sending request to {endpoint}")

            # Assuming model is a list with one element, e.g., ['qwen2']
            chat_model = model[0] if isinstance(model, list) else model

            res = requests.post(
                endpoint,
                json={
                    "model": chat_model,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                },
            )

            logger.info(f"Response: {res.json()}")

            if "error" in res.json():
                logger.error(res.json())
                if res.json()["error"]["type"] == "invalid_request_error":
                    logger.info("Input + output is longer than max_position_id.")
                    return None

            output = res.json()["choices"][0]["message"]["content"]

            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:

        return output

    output = output.strip()

    logger.info(f"Output: `{output[:20]}...`.")

    return output


def generate_together_stream(
    model,
    messages,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    api_base=API_BASE,
    api_key=API_KEY
):
    # endpoint = f"{api_base}/chat/completions"
    endpoint = api_base
    client = openai.OpenAI(api_key=api_key, base_url=endpoint)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature if temperature > 1e-4 else 0,
        max_tokens=max_tokens,
        stream=True,  # this time, we set stream=True
    )

    return response


def generate_openai(
    model,
    messages,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
):

    client = openai.OpenAI(
        base_url=API_BASE_2,
        api_key=API_KEY_2,
    )

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()

    return output


def inject_references_to_messages(
    messages,
    references,
):

    messages = copy.deepcopy(messages)

    system = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

    for i, reference in enumerate(references):

        system += f"\n{i+1}. {reference}"

    # if messages[0]["role"] == "system":

    #     messages[0]["content"] += "\n\n" + system

    # else:

    messages = [{"role": "system", "content": system}] + messages

    return messages


def generate_with_references(
    model,
    messages,
    references=[],
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    generate_fn=generate_together,
    api_base=API_BASE,
    api_key=API_KEY
):

    if len(references) > 0:

        messages = inject_references_to_messages(messages, references)

    return generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        api_base=api_base,
        api_key=api_key
    )
