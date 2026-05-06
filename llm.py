"""OpenAI-compatible LLM helpers for the IDEN-VLN generation pipeline."""

import base64
import requests

def _api_base(args):
    return getattr(args, 'API_BASE', 'https://api.openai.com/v1').rstrip('/')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def prompt_make(prompt_path, ex_prompt):    # set prompt
    with open(prompt_path, "r", encoding='utf-8') as f:  # open the file
        txt = f.readlines()
        prompt_system = txt[1]
        prompt = txt[3]  # read the user line
        if len(txt) > 4:
            for i in range(4, len(txt)):
                prompt = prompt + txt[i]
        prompt = prompt + ex_prompt
        # print(prompt_system)
        # print(prompt)
        return prompt_system, prompt


def llm(args, prompt_path, ex_prompt):
    """Call the main OpenAI-compatible chat model."""
    api_key = args.API_KEY

    prompt_system, prompt = prompt_make(prompt_path, ex_prompt)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": getattr(args, "MODEL_MAIN", "Qwen2.5-7B-Instruct"),
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": getattr(args, "TEMPERATURE", 0.2),
        "max_tokens": getattr(args, "MAX_TOKENS", 1024)
    }
    response = requests.post(f"{_api_base(args)}/chat/completions",
                            headers=headers, json=payload, timeout=60)

    output = response.json()
    # print(output["usage"])
    # print(output["choices"][0]['message'])
    return output["choices"][0]['message']["content"]


def llm_step(args, prompt_path, ex_prompt):
    """Call the step-level OpenAI-compatible chat model."""
    api_key = args.API_KEY

    prompt_system, prompt = prompt_make(prompt_path, ex_prompt)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": getattr(args, "MODEL_STEP", "Qwen2.5-7B-Instruct"),
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": getattr(args, "TEMPERATURE", 0.2),
        "max_tokens": getattr(args, "MAX_TOKENS", 1024)
    }
    response = requests.post(f"{_api_base(args)}/chat/completions",
                            headers=headers, json=payload, timeout=60)

    output = response.json()
    # print(output["usage"])
    # print(output["choices"][0]['message'])
    return output["choices"][0]['message']["content"]


def vision_llm(args, prompt_path, ex_prompt, img_path):
    """Call an OpenAI-compatible vision-language model."""
    base64_image = encode_image(img_path)
    api_key = args.API_KEY

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    prompt_system, prompt = prompt_make(prompt_path, ex_prompt)
    payload = {
        "model": getattr(args, "MODEL_VISION", "Qwen2.5-7B-Instruct"),
        "messages": [...],
        "temperature": getattr(args, "TEMPERATURE", 0.2),
        "max_tokens": getattr(args, "MAX_TOKENS", 1024)
    }
    response = requests.post(f"{_api_base(args)}/chat/completions",
                            headers=headers, json=payload, timeout=60)

    output = response.json()

    # print(output["usage"])
    # print(output["choices"][0]['message'])
    return output["choices"][0]['message']["content"]
