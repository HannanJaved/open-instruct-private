import os
import re

# Root directory to search
root_dir = '/data/horse/ws/hama901h-BFTranslation/lm_eval_results'
pattern = re.compile(r"huggingface_hub\.errors\.HfHubHTTPError: 429 Client Error: Too Many Requests for url")

result = []

for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.err'):
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if pattern.search(line):
                            # Get the relative path after lm_eval_results/
                            rel_path = os.path.relpath(file_path, root_dir)
                            result.append(rel_path)
                            break
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Output results
with open('/data/horse/ws/hama901h-BFTranslation/hf_429_err_files_script.txt', 'w') as out:
    for path in result:
        out.write(path + '\n')

print(f"Found {len(result)} files with 429 error.")
