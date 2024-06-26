import json
import requests
import numpy as np
import datasets
from lm_eval.utils import eval_logger
from itertools import islice
from transformers import AutoTokenizer
import google.generativeai as genai
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# APIキーの準備
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")

# Assume openai>=1.0.0
from openai import OpenAI

# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
)

# プロンプトテンプレートの準備
prompt_filename = "/content/mergekit-evolve-geniac/eval_tasks/prompt_eval_llamacpp.txt"
with open(prompt_filename, encoding='utf-8') as f:
    template_prompt = f.read()

import re

def extract_number_from_response(response):
    # Define a function to normalize full-width to half-width characters
    def normalize_full_width(s):
        return s.translate(str.maketrans(
            '０１２３４５６７８９',  # Full-width characters
            '0123456789'  # Half-width characters
        ))
    
    # Extract the number using a regular expression
    match = re.search(r'\d+', response)
    if match:
        num_str = match.group()
        # Normalize the number string to half-width characters
        num_str = normalize_full_width(num_str)
        # Convert the string to an integer
        num = int(num_str)
        return num
    else:
        return None
 
#ChatNTQ用のプロンプト
def build_prompt(user_query):
    sys_msg = "あなたは日本語で回答するChatbotです。"
    template = """
    <|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
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

    print(prompt)

    if pred == "":
        return 1

    # 評価
    chat_completion = openai.chat.completions.create(
    model="mistralai/Mixtral-8x22B-Instruct-v0.1",
    messages=[{"role": "system", "content": "あなたは数字のみで回答するAI採点ボットです。"},
     {"role": "user", "content": prompt}],
    temperature=0.5,#temperature to use for sampling. 0 means the output is deterministic. Values greater than 1 encourage more diversity
    top_p=0.9,#0 < top_p ≤ 1 Sample from the set of tokens with highest probability such that sum of probabilies is higher than p. Lower values focus on the most probable tokens.Higher values sample more low-probability tokens
    max_tokens=1024
    )

    for i in range(5):
        try:
            response = chat_completion.choices[0].message.content
            print("response: ", response)
            #response = gemini_model.generate_content(prompt)
            num = extract_number_from_response(response)
            if num is not None and 1 <= num <= 5:
                print("success: ", num)
                return num
        except Exception as e:
            print("error", response, e)
    ### 5回のRetryに失敗した場合、1を返す
    return 1

# スコアの計算
def process_results(doc, results):
    score = evaluate(results[0], doc["input"], doc["output"], doc["eval_aspect"])
    return {"acc": score}
