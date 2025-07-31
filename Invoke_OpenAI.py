# library
import os
import openai
import logging
from dotenv import load_dotenv
from logging_info import logging_setup

logger = logging_setup()

def get_open_ai_response(OpenAI_Key:str,prompt:str):
    try:
        openai.api_key = OpenAI_Key
        completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You're a helpful assistant who looks answers up for a user in a textbook and returns the answer to the user's question. If the answer is not in the textbook, you say 'I'm sorry, I don't have access to that information."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=5000,
        top_p=1
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.log(logging.ERROR,f"Having following issue with get_open_ai_response method - {e}")
 

