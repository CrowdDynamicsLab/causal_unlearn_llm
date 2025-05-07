import os
import yaml
import json
import argparse
import csv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

def load_prompt(prompt_file_path):
    """Load system and user prompts from a YAML file."""
    with open(prompt_file_path, 'r') as f:
        prompt_data = yaml.safe_load(f)
    return prompt_data['system_prompt'], prompt_data['user_prompt']

def contains_refusal_keywords(text):
    """Check if the response contains refusal keywords."""
    refusal_keywords = [
        "i'm sorry", "i am sorry", "ai model", "language model", "ai assistant",
        "cannot", "unable to", "as an ai", "ethical", "inappropriate",
        "not programmed", "against my", "policy", "guidelines"
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in refusal_keywords)

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def get_llm_response(client, model, system_prompt, formatted_user_prompt):
    """Get response from LLM with exponential backoff retry."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_user_prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

def process_csv_file(input_csv_file, output_jsonl_file, model_name, system_prompt, user_prompt_template, openai_api_key, stem_col_name, fragment_col_name):
    """Process the input CSV file and write results to an output JSONL file."""
    client = OpenAI(api_key=openai_api_key)
    
    results = []
    rows_to_process = []
    with open(input_csv_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        if stem_col_name not in reader.fieldnames:
            raise ValueError(f"Stem column '{stem_col_name}' not found in CSV. Available columns: {reader.fieldnames}")
        if fragment_col_name not in reader.fieldnames:
            raise ValueError(f"Fragment column '{fragment_col_name}' not found in CSV. Available columns: {reader.fieldnames}")
        for row in reader:
            rows_to_process.append(row)

    for row in tqdm(rows_to_process, desc="Processing CSV rows"):
        stem = row[stem_col_name]
        fragment = row[fragment_col_name]
        
        formatted_user_prompt = user_prompt_template.format(stem, fragment)
        
        llm_response_content = ""
        # Try up to 3 times if we get refusal responses
        for attempt in range(3):
            llm_response_content = get_llm_response(client, model_name, system_prompt, formatted_user_prompt)
            
            if not contains_refusal_keywords(llm_response_content):
                break
            
            print(f"Refusal detected for stem '{stem[:50]}...', retrying ({attempt+1}/3)")
            if attempt == 2:
                print("Max retries reached, keeping last (refusal) response.")
        
        reasoning = llm_response_content
        rating = ""
        if "Rating: " in llm_response_content:
            parts = llm_response_content.split("Rating: ", 1)
            reasoning = parts[0].strip()
            rating = parts[1].strip()
        
        results.append({
            "input_stem": stem,
            "input_fragment": fragment,
            "llm_full_response": llm_response_content,
            "llm_reasoning": reasoning,
            "llm_rating": rating
        })
    
    # Write results to output file
    with open(output_jsonl_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Evaluate sentence fluency using LLM based on CSV input.')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI model to use for evaluation.')
    parser.add_argument('--prompt_file', type=str, default='fluency_prompt.yaml', help='Path to YAML file with system and user prompts.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input CSV file.')
    parser.add_argument('--output_suffix', type=str, default='_fluency_judged', help='Suffix for the output JSONL file.')
    parser.add_argument('--openai_key_path', type=str, default='../openai_api.txt', help='Path to file containing OpenAI API key.')
    parser.add_argument('--stem-column', type=str, default='stem', help='Name of the column in CSV containing the sentence stem.')
    parser.add_argument('--fragment-column', type=str, default='fragment', help='Name of the column in CSV containing the sentence fragment.')
    
    args = parser.parse_args()

    try:
        with open(args.openai_key_path, 'r') as f:
            openai_key = f.read().strip()
    except FileNotFoundError:
        print(f"Error: OpenAI API key file not found at {args.openai_key_path}")
        return
    except Exception as e:
        print(f"Error reading OpenAI API key: {e}")
        return

    try:
        system_prompt, user_prompt_template = load_prompt(args.prompt_file)
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {args.prompt_file}")
        return
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return
        
    base_name, ext = os.path.splitext(args.input_file)
    # Ensure the extension is .jsonl for the output
    output_file_path = f"{base_name}{args.output_suffix}.jsonl"
    
    try:
        process_csv_file(args.input_file, output_file_path, args.model, system_prompt, user_prompt_template, openai_key, args.stem_column, args.fragment_column)
        print(f"Fluency evaluation complete. Results saved to {output_file_path}")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {args.input_file}")
    except ValueError as ve:
        print(f"Configuration error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")

if __name__ == "__main__":
    main() 