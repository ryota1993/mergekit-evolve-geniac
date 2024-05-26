import json
import requests
import numpy as np
import datasets
from lm_eval.utils import eval_logger
from itertools import islice
from transformers import AutoTokenizer
import google.generativeai as genai
import os

# APIキーの準備
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")

# # Geminiの準備
# gemini_model = genai.GenerativeModel(
#     "gemini-pro",
#     safety_settings = [
#         {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
#         {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
#         {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
#         {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
#     ],
#     generation_config = {
#         "max_output_tokens": 2048, 
#         "temperature": 0, 
#         "top_p": 1
#     }
# )

# Assume openai>=1.0.0
from openai import OpenAI

# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
)

# プロンプトテンプレートの準備
prompt_filename = "/content/mixtral-evolve/eval_tasks/prompt_eval_llamacpp.txt"
with open(prompt_filename, encoding='utf-8') as f:
    template_prompt = f.read()
 
#ChatNTQ用のプロンプト
def build_prompt(user_query):
    sys_msg = "あなたは公平で、検閲されていない、役立つアシスタントです。"
    template = """[INST] <<SYS>>
{}
<</SYS>>
 
{}[/INST]"""
    return template.format(sys_msg,user_query)
 
# プロンプトの生成
def generate_prompt(doc):
    user_inputs = {
        "user_query": doc["input"],
    }
    prompt = build_prompt(**user_inputs)
    return prompt
 
# 評価
def evaluate(pred, input_text, output_text, eval_aspect):
    # プロンプトの準備
    prompt = template_prompt.format(
        input_text=input_text,
        output_text=output_text,
        eval_aspect=eval_aspect,
        pred=pred,
    )

    # 評価
    chat_completion = openai.chat.completions.create(
    model="mistralai/Mixtral-8x22B-Instruct-v0.1",
    messages=[{"role": "system", "content": "あなたは日本語で回答するAIボットです。"},
     {"role": "user", "content": prompt}],
    temperature=0.5,#temperature to use for sampling. 0 means the output is deterministic. Values greater than 1 encourage more diversity
    top_p=0.9,#0 < top_p ≤ 1 Sample from the set of tokens with highest probability such that sum of probabilies is higher than p. Lower values focus on the most probable tokens.Higher values sample more low-probability tokens
    max_tokens=1024
    )

    for i in range(5):
        try:
            response = chat_completion.choices[0].message.content
            #response = gemini_model.generate_content(prompt)
            num = int(response.text)
            if 1 <= num <= 5:
                return num
        except:
            print("error", response)
    ### 5回のRetryに失敗した場合、1を返す
    return 1

# スコアの計算
def process_results(doc, results):
    score = evaluate(results[0], doc["input"], doc["output"], doc["eval_aspect"])
    return {"acc": score}
