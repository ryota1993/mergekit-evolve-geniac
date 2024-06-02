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

# Assume openai>=1.0.0
from openai import OpenAI

# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
)

# プロンプトテンプレートの準備
prompt_filename = "/content/mergekit-evolve-geniac/eval_tasks/prompt_eval_math.txt"
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
 
# 評価
def evaluate(pred, question, answer):
    # プロンプトの準備
    prompt = template_prompt.format(
        question=question,
        answer=answer,
        pred=pred
    )

    print(prompt)

    # 評価
    chat_completion = openai.chat.completions.create(
    model="mistralai/Mixtral-8x22B-Instruct-v0.1",
    messages=[{"role": "system", "content": "あなたは1から5までの数字のみで回答するAI採点ボットです。"},
     {"role": "user", "content": prompt}],
    temperature=0.5,#temperature to use for sampling. 0 means the output is deterministic. Values greater than 1 encourage more diversity
    top_p=0.9,#0 < top_p ≤ 1 Sample from the set of tokens with highest probability such that sum of probabilies is higher than p. Lower values focus on the most probable tokens.Higher values sample more low-probability tokens
    max_tokens=1
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
    score = evaluate(results[0], doc["question"], doc["answer_number"])
    return {"acc": score}