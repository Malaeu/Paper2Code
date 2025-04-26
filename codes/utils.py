import json
import re
import os

def extract_planning(trajectories_json_file_path):
    with open(trajectories_json_file_path) as f:
        traj = json.load(f)

    context_lst = []
    for turn in traj:
        if turn['role'] == 'assistant':
            # context_lst.append(turn['content'])
            content = turn['content']
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            context_lst.append(content)


    context_lst = context_lst[:3] 

    return context_lst



def content_to_json(data):
    clean_data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()

    clean_data = re.sub(r'(".*?"),\s*#.*', r'\1,', clean_data)

    clean_data = re.sub(r',\s*\]', ']', clean_data)

    clean_data = re.sub(r'\n\s*', '', clean_data)


    # JSON parsing
    try:
        json_data = json.loads(clean_data)
        return json_data
    except json.JSONDecodeError as e:
        # print(e)
        return content_to_json2(data)
        
    
def content_to_json2(data):
    # remove [CONTENT][/CONTENT]
    clean_data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()

    # "~~~~", #comment -> "~~~~",
    clean_data = re.sub(r'(".*?"),\s*#.*', r'\1,', clean_data)

    # "~~~~" #comment → "~~~~"
    clean_data = re.sub(r'(".*?")\s*#.*', r'\1', clean_data)


    # ("~~~~",] -> "~~~~"])
    clean_data = re.sub(r',\s*\]', ']', clean_data)

    clean_data = re.sub(r'\n\s*', '', clean_data)

    # JSON parsing
    try:
        json_data = json.loads(clean_data)
        return json_data
    
    except json.JSONDecodeError as e:
        # print("Json parsing error", e)
        return content_to_json3(data)

def content_to_json3(data):
    # remove [CONTENT] [/CONTENT]
    clean_data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()

    # "~~~~", #comment -> "~~~~",
    clean_data = re.sub(r'(".*?"),\s*#.*', r'\1,', clean_data)

    # "~~~~" #comment → "~~~~"
    clean_data = re.sub(r'(".*?")\s*#.*', r'\1', clean_data)

    # remove ("~~~~",] -> "~~~~"])
    clean_data = re.sub(r',\s*\]', ']', clean_data)

    clean_data = re.sub(r'\n\s*', '', clean_data) 
    clean_data = re.sub(r'"""', '"', clean_data)  # Replace triple double quotes
    clean_data = re.sub(r"'''", "'", clean_data)  # Replace triple single quotes
    clean_data = re.sub(r"\\", "'", clean_data)  # Replace \ 

    # JSON parsing
    try:
        json_data = json.loads(f"""{clean_data}""")
        return json_data
    
    except json.JSONDecodeError as e:
        print(e)
        
        # print(f"[DEBUG] utils.py > content_to_json3 ")
        return None 
    


def extract_code_from_content(content):
    pattern = r'^```(?:\w+)?\s*\n(.*?)(?=^```)```'
    code = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
    if len(code) == 0:
        return ""
    else:
        return code[0]
    

def format_json_data(data):
    formatted_text = ""
    for key, value in data.items():
        formatted_text += "-" * 40 + "\n"
        formatted_text += "[" + key + "]\n"
        if isinstance(value, list):
            for item in value:
                formatted_text += f"- {item}\n"
        else:
            formatted_text += str(value) + "\n"
        formatted_text += "\n"
    return formatted_text


def cal_cost(response_json, model_name):
    model_cost = {
        # gpt-4.1
        "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
        "gpt-4.1-2025-04-14": {"input": 2.00, "cached_input": 0.50, "output": 8.00},

        # gpt-4.1-mini
        "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
        "gpt-4.1-mini-2025-04-14": {"input": 0.40, "cached_input": 0.10, "output": 1.60},

        # gpt-4.1-nano
        "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
        "gpt-4.1-nano-2025-04-14": {"input": 0.10, "cached_input": 0.025, "output": 0.40},

        # gpt-4.5-preview
        "gpt-4.5-preview": {"input": 75.00, "cached_input": 37.50, "output": 150.00},
        "gpt-4.5-preview-2025-02-27": {"input": 75.00, "cached_input": 37.50, "output": 150.00},

        # gpt-4o
        "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
        "gpt-4o-2024-08-06": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
        "gpt-4o-2024-11-20": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
        "gpt-4o-2024-05-13": {"input": 5.00, "cached_input": None, "output": 15.00},

        # gpt-4o-audio-preview
        "gpt-4o-audio-preview": {"input": 2.50, "cached_input": None, "output": 10.00},
        "gpt-4o-audio-preview-2024-12-17": {"input": 2.50, "cached_input": None, "output": 10.00},
        "gpt-4o-audio-preview-2024-10-01": {"input": 2.50, "cached_input": None, "output": 10.00},

        # gpt-4o-realtime-preview
        "gpt-4o-realtime-preview": {"input": 5.00, "cached_input": 2.50, "output": 20.00},
        "gpt-4o-realtime-preview-2024-12-17": {"input": 5.00, "cached_input": 2.50, "output": 20.00},
        "gpt-4o-realtime-preview-2024-10-01": {"input": 5.00, "cached_input": 2.50, "output": 20.00},

        # gpt-4o-mini
        "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
        "gpt-4o-mini-2024-07-18": {"input": 0.15, "cached_input": 0.075, "output": 0.60},

        # gpt-4o-mini-audio-preview
        "gpt-4o-mini-audio-preview": {"input": 0.15, "cached_input": None, "output": 0.60},
        "gpt-4o-mini-audio-preview-2024-12-17": {"input": 0.15, "cached_input": None, "output": 0.60},

        # gpt-4o-mini-realtime-preview
        "gpt-4o-mini-realtime-preview": {"input": 0.60, "cached_input": 0.30, "output": 2.40},
        "gpt-4o-mini-realtime-preview-2024-12-17": {"input": 0.60, "cached_input": 0.30, "output": 2.40},

        # o1
        "o1": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
        "o1-2024-12-17": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
        "o1-preview-2024-09-12": {"input": 15.00, "cached_input": 7.50, "output": 60.00},

        # o1-pro
        "o1-pro": {"input": 150.00, "cached_input": None, "output": 600.00},
        "o1-pro-2025-03-19": {"input": 150.00, "cached_input": None, "output": 600.00},

        # o3
        "o3": {"input": 10.00, "cached_input": 2.50, "output": 40.00},
        "o3-2025-04-16": {"input": 10.00, "cached_input": 2.50, "output": 40.00},

        # o4-mini
        "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40},
        "o4-mini-2025-04-16": {"input": 1.10, "cached_input": 0.275, "output": 4.40},

        # o3-mini
        "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
        "o3-mini-2025-01-31": {"input": 1.10, "cached_input": 0.55, "output": 4.40},

        # o1-mini
        "o1-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
        "o1-mini-2024-09-12": {"input": 1.10, "cached_input": 0.55, "output": 4.40},

        # gpt-4o-mini-search-preview
        "gpt-4o-mini-search-preview": {"input": 0.15, "cached_input": None, "output": 0.60},
        "gpt-4o-mini-search-preview-2025-03-11": {"input": 0.15, "cached_input": None, "output": 0.60},

        # gpt-4o-search-preview
        "gpt-4o-search-preview": {"input": 2.50, "cached_input": None, "output": 10.00},
        "gpt-4o-search-preview-2025-03-11": {"input": 2.50, "cached_input": None, "output": 10.00},

        # computer-use-preview
        "computer-use-preview": {"input": 3.00, "cached_input": None, "output": 12.00},
        "computer-use-preview-2025-03-11": {"input": 3.00, "cached_input": None, "output": 12.00},

        # gpt-image-1
        "gpt-image-1": {"input": 5.00, "cached_input": None, "output": None},
    }

    
    prompt_tokens = response_json["usage"]["prompt_tokens"]
    completion_tokens = response_json["usage"]["completion_tokens"]
    cached_tokens = response_json["usage"]["prompt_tokens_details"].get("cached_tokens", 0)

    # input token = (prompt_tokens - cached_tokens)
    actual_input_tokens = prompt_tokens - cached_tokens
    output_tokens = completion_tokens

    cost_info = model_cost[model_name]

    input_cost = (actual_input_tokens / 1_000_000) * cost_info['input']
    cached_input_cost = (cached_tokens / 1_000_000) * cost_info['cached_input']
    output_cost = (output_tokens / 1_000_000) * cost_info['output']

    total_cost = input_cost + cached_input_cost + output_cost

    return {
        'model_name': model_name,
        'actual_input_tokens': actual_input_tokens,
        'input_cost': input_cost,
        'cached_tokens': cached_tokens,
        'cached_input_cost': cached_input_cost,
        'output_tokens': output_tokens,
        'output_cost': output_cost,
        'total_cost': total_cost,
    }

def load_accumulated_cost(accumulated_cost_file):
    if os.path.exists(accumulated_cost_file):
        with open(accumulated_cost_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("total_cost", 0.0)
    else:
        return 0.0

def save_accumulated_cost(accumulated_cost_file, cost):
    with open(accumulated_cost_file, "w", encoding="utf-8") as f:
        json.dump({"total_cost": cost}, f)

def print_response(completion_json):
    print("============================================")
    print(completion_json['choices'][0]['message']['content'])
    print("============================================\n")

def print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost):
    usage_info = cal_cost(completion_json, gpt_version)

    current_cost = usage_info['total_cost']
    total_accumulated_cost += current_cost

    output_lines = []
    output_lines.append("🌟 Usage Summary 🌟")
    output_lines.append(f"{current_stage}")
    output_lines.append(f"🛠️ Model: {usage_info['model_name']}")
    output_lines.append(f"📥 Input tokens: {usage_info['actual_input_tokens']} (Cost: ${usage_info['input_cost']:.8f})")
    output_lines.append(f"📦 Cached input tokens: {usage_info['cached_tokens']} (Cost: ${usage_info['cached_input_cost']:.8f})")
    output_lines.append(f"📤 Output tokens: {usage_info['output_tokens']} (Cost: ${usage_info['output_cost']:.8f})")
    output_lines.append(f"💵 Current total cost: ${current_cost:.8f}")
    output_lines.append(f"🪙 Accumulated total cost so far: ${total_accumulated_cost:.8f}")
    output_lines.append("============================================\n")

    output_text = "\n".join(output_lines)
    
    print(output_text)

    with open(f"{output_dir}/cost_info.log", "a", encoding="utf-8") as f:
        f.write(output_text + "\n")
    
    return total_accumulated_cost
