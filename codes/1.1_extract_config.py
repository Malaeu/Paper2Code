import json
import re
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--paper_name',type=str)
parser.add_argument('--output_dir',type=str, default="")

args    = parser.parse_args()

output_dir = args.output_dir

with open(f'{output_dir}/planning_trajectories.json', encoding='utf8') as f:
    traj = json.load(f)

yaml_raw_content = ""
for turn_idx, turn in enumerate(traj):
        if turn_idx == 8:
            yaml_raw_content = turn['content']   

if "</think>" in yaml_raw_content:
    yaml_raw_content = yaml_raw_content.split("</think>")[-1]

match = re.search(r"```yaml\n(.*?)\n```", yaml_raw_content, re.DOTALL)
if match:
    yaml_content = match.group(1)
    with open(f'{output_dir}/planning_config.yaml', 'w', encoding='utf8') as f:
        f.write(yaml_content)
else:
    # print("No YAML content found.")
    match2 = re.search(r"```yaml\\n(.*?)\\n```", yaml_raw_content, re.DOTALL)
    if match2:
        yaml_content = match2.group(1)
        with open(f'{output_dir}/planning_config.yaml', 'w', encoding='utf8') as f:
            f.write(yaml_content)
    else:
        print("No YAML content found.")