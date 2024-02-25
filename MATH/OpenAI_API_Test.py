from time import sleep
import os
from dotenv import load_dotenv
import sys
import guidance
from guidance import models, gen
from guidance import select
from guidance import user, assistant, system
import tiktoken
from tiktoken import encoding_name_for_model

main_dir = os.path.abspath(__file__)
for i in range(0, 1):
    main_dir = os.path.dirname(main_dir)
print(main_dir)
os.chdir(main_dir)
sys.path.append(main_dir)
load_dotenv()
apikey = os.getenv('OPENAI_API_KEY')

import openai

# Testing for guidance and OpenAI API.
lm = models.OpenAIChat(model="gpt4-1106-preview", base_url="https://drchat.xyz", api_key=apikey, tokenizer=tiktoken.get_encoding("cl100k_base"))
with system():
    lm += "You are a math expert."

with user():
    lm += "The area of a circle with a radius of 1 cm is 2*pi cm^2. Is this true or false. You answer options are [TRUE, FALSE] Answer in one word."

with assistant():
    lm += select(["TRUE", "FALSE"], "answer")

print("\n")
print(lm["answer"])