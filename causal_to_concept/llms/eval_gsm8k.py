"""
Evaluate GSM8K performance for:
  1) pretrained base model
  2) pretrained base model + runtime head interventions (baukit TraceDict)
  3) fine-tuned model (weights changed)

This script is intentionally self-contained and writes outputs to a local folder.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from einops import rearrange

try:
    from baukit import TraceDict
except Exception as e:  # pragma: no cover
    TraceDict = None  # type: ignore[assignment]
    _BAUKIT_IMPORT_ERROR = e

try:
    from sentence_transformers import SentenceTransformer
    sentence_embedding = SentenceTransformer('all-MiniLM-L6-v2')
    _SENTENCE_EMBEDDING_AVAILABLE = True
except Exception as e:
    sentence_embedding = None
    _SENTENCE_EMBEDDING_AVAILABLE = False
    _SENTENCE_EMBEDDING_ERROR = e


_FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?[\d,]+(?:\.\d+)?)")
_FALLBACK_NUMBER_RE = re.compile(r"([-+]?\d+)")

# Model name mapping (similar to validate_2fold_toxic.py)
HF_NAMES = {
    'llama_1B': 'meta-llama/Llama-3.2-1B',
    'llama_3B': 'meta-llama/Llama-3.2-3B',
    'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'alpaca_7B': 'circulus/alpaca-7b',
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b',
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
    'vicuna_13B': 'lmsys/vicuna-13b-v1.5',
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'gemma3_4B': 'google/gemma-3-4b-it',
    'mistral_7B': 'mistralai/Mistral-7B-v0.1',
    'mistral_7B_Instruct': 'mistralai/Mistral-7B-Instruct-v0.2',
    'qwen_7B': 'Qwen/Qwen2.5-7B-Instruct',
}

# --- ADDED: Standard 8-Shot CoT Examples from Wei et al. (2022) ---
GSM8K_COT_EXAMPLES = """
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and his dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Question: There were 9 computers in the server room. Five more computers were installed each day for 4 days. How many computers are now in the server room?
Answer: There were originally 9 computers. For 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is 8.
"""
# ------------------------------------------------------------------


@dataclass(frozen=True)
class GSM8KExample:
    idx: int
    question: str
    answer: str


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_gsm8k(split: str, max_examples: Optional[int]) -> List[GSM8KExample]:
    ds = load_dataset("gsm8k", "main", split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    examples: List[GSM8KExample] = []
    for i, item in enumerate(ds):
        examples.append(
            GSM8KExample(
                idx=i,
                question=item["question"],
                answer=item["answer"],
            )
        )
    return examples


# def _format_prompt(question: str) -> str:
#     """
#     Using standard 8-shot Chain-of-Thought (CoT) prompting.
#     This creates an in-context learning setup where the model sees 8 examples
#     of (Question -> Step-by-step Reasoning -> Answer) before the target question.
#     """
#     return (
#         f"{GSM8K_COT_EXAMPLES.strip()}\n\n"
#         f"Q: {question}\n"
#         "A:"
#     )

def _format_prompt(question: str, tokenizer: AutoTokenizer, is_instruct: bool) -> str:
    if is_instruct and tokenizer.chat_template is not None:
        # NEW: Uses chat template for Mistral/Llama-Instruct
        messages = [{"role": "user", "content": f"... {question} ..."}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback for Base models
        return f"{GSM8K_COT_EXAMPLES.strip()}\n\nQ: {question}\nA:"


def _extract_answer(completion: str) -> Optional[str]:
    """
    Extracts the numeric answer from the completion.
    First looks for 'The answer is <number>', then looks for the last number.
    """
    # 1. Try to find "The answer is X" which is standard for CoT
    match = re.search(r"The answer is ([-+]?\d+(?:\.\d+)?)", completion)
    if match:
        return match.group(1)
    
    # 2. Fallback: look for #### pattern if model happened to output it
    match = _FINAL_ANSWER_RE.search(completion)
    if match:
        return match.group(1)

    # 3. Fallback: find the last number in the text
    matches = _FALLBACK_NUMBER_RE.findall(completion)
    if matches:
        return matches[-1]
        
    return None


def _is_correct(predicted: Optional[str], gold: str) -> bool:
    if predicted is None:
        return False
    # Gold answer in GSM8K dataset usually ends with "#### <number>"
    # We need to extract that number.
    gold_match = _FINAL_ANSWER_RE.search(gold)
    if not gold_match:
        # Fallback if the dataset format is different than expected
        gold_clean = gold.strip().split()[-1]
    else:
        gold_clean = gold_match.group(1)

    try:
        # Compare as floats to handle potential formatting differences (e.g. 1 vs 1.0)
        return abs(float(predicted.replace(",", "")) - float(gold_clean.replace(",", ""))) < 1e-6
    except ValueError:
        return False


def _load_interventions(path: str) -> Dict[str, Any]:
    """Load interventions from .npy or .pkl file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        # Load numpy array - this contains directions indexed by flattened (layer, head) indices
        data = np.load(path, allow_pickle=True)
        # If it's a numpy array (not a dict saved as .npy), return it with a special key
        if isinstance(data, np.ndarray):
            # Check if it's a scalar array containing a dict (saved with allow_pickle=True)
            if data.dtype == object and data.ndim == 0:
                return data.item()
            else:
                # Regular numpy array - return with special key
                return {"directions_array": data}
        else:
            # Not a numpy array (shouldn't happen with np.load, but handle it)
            return data
    elif ext in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        # Try pickle as fallback
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            raise ValueError(f"Unsupported interventions file format: {ext}. Expected .npy or .pkl")


def _layer_head_to_flattened_idx(layer: int, head: int, num_heads: int) -> int:
    """Convert (layer, head) to flattened index."""
    return layer * num_heads + head


def _load_heads_file(path: str, num_heads: int) -> List[Tuple[int, int]]:
    """Load heads file (.npy) and return top num_heads as list of (layer, head) tuples."""
    heads_array = np.load(path)
    # Ensure we have the right shape
    if heads_array.ndim == 2 and heads_array.shape[1] == 2:
        # Array of [layer, head] pairs
        heads_list = [(int(l), int(h)) for l, h in heads_array[:num_heads]]
    elif heads_array.ndim == 1:
        # Flattened indices - would need num_heads to convert, but for now assume it's already pairs
        # This is a fallback - adjust based on your actual format
        heads_list = [(int(h // 32), int(h % 32)) for h in heads_array[:num_heads]]  # Assuming 32 heads per layer
    else:
        raise ValueError(f"Unexpected heads array shape: {heads_array.shape}")
    return heads_list


def _resolve_model_name(model_input: str) -> Tuple[str, str]:
    """
    Resolves a model name input (e.g., 'llama3_8B') to the actual HuggingFace path.
    Returns (resolved_path, clean_model_name) where clean_model_name is used for file naming.
    """
    # If it's already a full path or HF model ID, use it as-is
    if '/' in model_input or model_input.startswith('meta-llama/') or model_input.startswith('mistralai/') or model_input.startswith('Qwen/'):
        # Extract a clean name from the path
        clean_name = model_input.split('/')[-1].replace('-', '_').replace('.', '_')
        return model_input, clean_name
    
    # Check special cases first
    if model_input == 'llama3_8B':
        return 'meta-llama/Meta-Llama-3-8B', 'llama3_8B'
    elif model_input == 'vicuna_13B':
        return 'lmsys/vicuna-13b-v1.5', 'vicuna_13B'
    
    # Check HF_NAMES mapping
    if model_input in HF_NAMES:
        return HF_NAMES[model_input], model_input
    
    # Fallback: assume it's already a valid path/name
    return model_input, model_input.replace('/', '_').replace('-', '_')


def _eval_variant(
    variant_name: str,
    model_path: str,
    model_name: str,  # Clean model name for file naming
    examples: List[GSM8KExample],
    out_dir: str,
    dtype: torch.dtype,
    device_map: str,
    local_files_only: bool,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    interventions_path: Optional[str] = None,
    heads_path: Optional[str] = None,
    num_heads: Optional[int] = None,
    alpha: float = 1.0,
    edit_mode: bool = False,
    prompt_encoding_text: str = "full_prompt",
) -> Dict[str, Any]:
    
    print(f"\n--- Evaluating variant: {variant_name} ---")
    print(f"Model path: {model_path}")
    
    # Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        local_files_only=local_files_only,
    )
    model.eval()

    # Setup Interventions if needed
    interventions_dict = {}
    intervention_fn = None
    layers_to_intervene = []
    
    if interventions_path is not None:
        if heads_path is None or num_heads is None:
            raise ValueError("When using interventions, both --heads_path and --num_heads must be provided")
    
    if interventions_path and heads_path and num_heads is not None:
        if TraceDict is None:
             raise ImportError(f"baukit not installed. {_BAUKIT_IMPORT_ERROR}")
        
        # Load heads to intervene
        top_heads = _load_heads_file(heads_path, num_heads)
        print(f"Loaded {len(top_heads)} heads to intervene from {heads_path}")
        print(f"Top heads: {top_heads[:5]}...")  # Show first 5
        
        # Load interventions data
        interventions_data = _load_interventions(interventions_path)
        print(f"Loaded interventions from {interventions_path}")
        
        # Get model config to determine num_heads
        try:
            model_num_heads = int(model.config.num_attention_heads)
        except Exception as e:
            raise RuntimeError("Could not determine num_attention_heads from model.config") from e
        
        if isinstance(interventions_data, dict):
            # Check if it contains directions_array (from .npy file)
            if "directions_array" in interventions_data:
                # Format: numpy array indexed by flattened (layer, head) indices
                directions_array = interventions_data["directions_array"]
                interventions_dict = {}
                for layer, head in top_heads:
                    layer_name = f"model.layers.{layer}.self_attn.o_proj"
                    if layer_name not in interventions_dict:
                        interventions_dict[layer_name] = []
                    
                    # Get direction from array using flattened index
                    flattened_idx = _layer_head_to_flattened_idx(layer, head, model_num_heads)
                    if flattened_idx < len(directions_array):
                        direction = directions_array[flattened_idx]
                        proj_val_std = 1.0
                        
                        # Convert to tensor if needed
                        if isinstance(direction, np.ndarray):
                            direction = torch.from_numpy(direction).float()
                        elif not isinstance(direction, torch.Tensor):
                            direction = torch.tensor(direction, dtype=torch.float32)
                        
                        # Normalize direction (only for 1D vectors)
                        if direction.ndim == 1:
                            direction = direction / (direction.norm() + 1e-8)
                        elif direction.ndim == 2:
                            # Matrix direction - keep as is
                            pass
                        
                        interventions_dict[layer_name].append((head, direction, proj_val_std))
                    else:
                        print(f"Warning: Flattened index {flattened_idx} out of range for layer {layer}, head {head}, skipping")
            # Check if it's already in the format {layer_name: [(head, direction, std), ...]}
            elif (sample_key := list(interventions_data.keys())[0] if interventions_data else None) and "model.layers" in str(sample_key) and ".self_attn.o_proj" in str(sample_key):
                # Already in correct format - filter to only include our selected heads
                interventions_dict = {}
                for layer, head in top_heads:
                    layer_name = f"model.layers.{layer}.self_attn.o_proj"
                    if layer_name in interventions_data:
                        # Find the entry for this head
                        for entry in interventions_data[layer_name]:
                            if isinstance(entry, (list, tuple)) and len(entry) >= 2 and entry[0] == head:
                                if layer_name not in interventions_dict:
                                    interventions_dict[layer_name] = []
                                interventions_dict[layer_name].append(entry)
                                break
            else:
                # Need to build from scratch
                interventions_dict = {}
                for layer, head in top_heads:
                    layer_name = f"model.layers.{layer}.self_attn.o_proj"
                    if layer_name not in interventions_dict:
                        interventions_dict[layer_name] = []
                    
                    # Try to find direction for this (layer, head) pair
                    direction = None
                    proj_val_std = 1.0
                    
                    # Format 1: {(layer, head): direction}
                    if (layer, head) in interventions_data:
                        direction = interventions_data[(layer, head)]
                    # Format 2: {layer_name: {head: direction}}
                    elif layer_name in interventions_data:
                        if isinstance(interventions_data[layer_name], dict) and head in interventions_data[layer_name]:
                            direction = interventions_data[layer_name][head]
                    # Format 3: directions stored as array indexed by flattened index (legacy format)
                    elif "directions" in interventions_data:
                        flattened_idx = _layer_head_to_flattened_idx(layer, head, model_num_heads)
                        directions = interventions_data["directions"]
                        if flattened_idx < len(directions):
                            direction = directions[flattened_idx]
                    
                    if direction is None:
                        print(f"Warning: No direction found for layer {layer}, head {head}, skipping")
                        continue
                    
                    # Convert to tensor if needed
                    if isinstance(direction, np.ndarray):
                        direction = torch.from_numpy(direction).float()
                    elif not isinstance(direction, torch.Tensor):
                        direction = torch.tensor(direction, dtype=torch.float32)
                    
                    # Normalize direction (only for 1D vectors)
                    if direction.ndim == 1:
                        direction = direction / (direction.norm() + 1e-8)
                    elif direction.ndim == 2:
                        # Matrix direction - keep as is
                        pass
                    
                    interventions_dict[layer_name].append((head, direction, proj_val_std))
        else:
            raise ValueError(f"Interventions data must be a dict, got {type(interventions_data)}")
        
        # Sort by head index for each layer
        for layer_name in interventions_dict:
            interventions_dict[layer_name] = sorted(interventions_dict[layer_name], key=lambda x: x[0])
        
        layers_to_intervene = list(interventions_dict.keys())
        print(f"Intervening on {len(layers_to_intervene)} layers with {sum(len(v) for v in interventions_dict.values())} total heads")
        
        # Create intervention function
        def create_intervention_fn(interventions: Dict[str, List[Tuple[int, torch.Tensor, float]]], 
                                   num_heads: int, alpha: float, prompt_encoding: Optional[np.ndarray] = None):
            def intervention_fn(head_output, layer_name):
                if layer_name not in interventions:
                    return head_output
                
                # Preserve original dtype and device
                original_dtype = head_output.dtype
                head_output = head_output.detach()
                bsz, seq_len, hidden = head_output.shape
                head_dim = hidden // num_heads
                
                # Reshape to [batch, seq_len, num_heads, head_dim]
                head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
                
                # Convert prompt_encoding to tensor if available
                prompt_encoding_tensor = None
                if prompt_encoding is not None:
                    assert prompt_encoding.shape == (384,), f"Expected prompt_encoding shape (384,), got {prompt_encoding.shape}"
                    prompt_encoding_tensor = torch.tensor(prompt_encoding, device=head_output.device, dtype=original_dtype).reshape(-1, 384)
                
                # Apply interventions for each head in this layer
                for head_idx, direction, proj_val_std in interventions[layer_name]:
                    if head_idx < 0 or head_idx >= num_heads:
                        continue
                    
                    # Move direction to same device and dtype as head_output
                    direction = direction.to(device=head_output.device, dtype=original_dtype)
                    
                    # Apply intervention: add alpha * proj_val_std * direction
                    # Only intervene on the last token position (start_edit_location == 'lt')
                    if direction.ndim == 1:
                        # Vector direction (center of mass or probe direction)
                        if direction.numel() != head_dim:
                            print(f"Warning: direction size {direction.numel()} != head_dim {head_dim} for {layer_name} head {head_idx}")
                            continue
                        assert proj_val_std is not None, f"proj_val_std must be provided for vector directions"
                        direction_to_add = direction.view(1, head_dim)  # [1, head_dim] for broadcasting
                        head_output[:, -1, head_idx, :] += (alpha * proj_val_std) * direction_to_add
                    elif direction.ndim == 2:
                        # Matrix direction (for special directions or matrix directions)
                        # direction shape: [head_dim, prompt_dim] for special, or [head_dim, head_dim] for matrix
                        # Following validate_2fold_toxic.py logic
                        if prompt_encoding_tensor is not None:
                            # use_special_direction: use prompt_encoding (matching validate_2fold_toxic.py line 359)
                            # uses batch size = 1
                            direction_to_add = torch.matmul(prompt_encoding_tensor, direction.T)  # [1, head_dim]
                        else:
                            # No prompt_encoding: use current activation (matching validate_2fold_toxic.py line 356)
                            current_activation = head_output[:, -1, head_idx, :]  # [batch, head_dim]
                            direction_to_add = torch.matmul(current_activation, direction.T)  # [batch, head_dim]
                        
                        # Normalize the resulting direction vector (matching validate_2fold_toxic.py line 360)
                        direction_norm = torch.linalg.norm(direction_to_add, dim=1, keepdim=True)  # [batch or 1, 1]
                        direction_to_add = direction_to_add / (direction_norm + 1e-8)
                        
                        # Compute proj_val_std online (matching validate_2fold_toxic.py lines 361-363)
                        # Note: We don't have tuning_activations in eval_gsm8k.py, so we use a default value
                        # In the original code: proj_vals = activations @ direction_to_add.T, then std
                        # Since we can't compute it online, we use proj_val_std from the stored value or default to 1.0
                        if proj_val_std is None:
                            # Default to 1.0 when not available (matrix directions typically have None)
                            proj_val_std_value = 1.0
                        else:
                            proj_val_std_value = proj_val_std
                        
                        # Convert to tensor with proper dtype and device
                        # PyTorch will broadcast scalar automatically: scalar * [batch, head_dim] -> [batch, head_dim]
                        proj_val_std_tensor = torch.tensor(proj_val_std_value, device=head_output.device, dtype=original_dtype)
                        
                        # Apply the intervention (matching validate_2fold_toxic.py line 382)
                        head_output[:, -1, head_idx, :] += alpha * proj_val_std_tensor * direction_to_add
                
                # Reshape back
                head_output = rearrange(head_output, 'b s h d -> b s (h d)')
                return head_output.to(dtype=original_dtype)
            
            return intervention_fn

    # Evaluation Loop
    predictions = []
    correct_count = 0
    is_instruct = "instruct" in model_path.lower() or "chat" in model_path.lower()
    
    for ex in tqdm(examples, desc=f"Eval {variant_name}"):
        prompt = _format_prompt(ex.question, tokenizer, is_instruct)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Compute prompt encoding if available and needed for special directions
        prompt_encoding = None
        if interventions_dict and _SENTENCE_EMBEDDING_AVAILABLE:
            # Check if we have any matrix directions (special directions)
            has_matrix_directions = any(
                direction.ndim == 2 
                for layer_name in interventions_dict 
                for _, direction, _ in interventions_dict[layer_name]
            )
            if has_matrix_directions:
                # Compute prompt encoding based on user preference
                if prompt_encoding_text == "full_prompt":
                    # Use the full formatted prompt (with CoT examples) - matches what's sent to model
                    text_to_encode = prompt
                else:  # question_only
                    # Use just the question text
                    text_to_encode = ex.question
                prompt_encoding = sentence_embedding.encode(text_to_encode)
        
        # Generate
        with torch.no_grad():
            if interventions_dict:
                # Create intervention function with prompt encoding for this example
                example_intervention_fn = create_intervention_fn(
                    interventions_dict, model_num_heads, alpha, prompt_encoding=prompt_encoding
                )
                # Apply interventions using TraceDict
                with TraceDict(model, layers_to_intervene, edit_output=example_intervention_fn):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        pad_token_id=tokenizer.eos_token_id
                    )
            else:
                # No interventions
                outputs = model.generate(
                    **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id
                )
            
        full_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the newly generated part
        generated_text = full_out[len(prompt):]
        
        predicted_answer = _extract_answer(generated_text)
        is_correct = _is_correct(predicted_answer, ex.answer)
        
        if is_correct:
            correct_count += 1
            
        predictions.append({
                        "idx": ex.idx,
                        "question": ex.question,
            "gold_answer": ex.answer,
            "full_output": full_out,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct
        })

    accuracy = correct_count / len(examples)
    print(f"Accuracy for {variant_name}: {accuracy:.2%}")
    
    # Save detailed results with model name in filename
    # Include alpha value if interventions are used
    _ensure_dir(out_dir)
    if interventions_path is not None:
        # Format alpha to avoid issues with decimal points in filenames
        alpha_str = str(alpha).replace('.', '_')
        out_file = os.path.join(out_dir, f"{model_name}_{variant_name}_alpha{alpha_str}.json")
    else:
        out_file = os.path.join(out_dir, f"{model_name}_{variant_name}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "predictions": predictions}, f, indent=2)
        
    return {
        "variant": variant_name,
        "model_name": model_name,
        "accuracy": accuracy,
        "output_file": out_file
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--finetuned_model", type=str, default=None)
    parser.add_argument("--interventions_path", type=str, default=None, help="Path to .npy or .pkl file with intervention directions")
    parser.add_argument("--heads_path", type=str, default=None, help="Path to .npy file with layer-head pairs to intervene on")
    parser.add_argument("--num_heads", type=int, default=None, help="Number of top heads to select from heads_path")
    parser.add_argument("--out_dir", type=str, default="results_gsm8k")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=256) # Increased for CoT
    parser.add_argument("--temperature", type=float, default=0.0)  # Greedy for eval
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--edit_mode", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--prompt_encoding_text", type=str, default="full_prompt", 
                       choices=["full_prompt", "question_only"],
                       help="What text to use for prompt encoding: 'full_prompt' (with CoT examples) or 'question_only' (just the question)")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(42)
    
    examples = _load_gsm8k("test", args.max_examples)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    results = []
    model_name_for_summary = None

    # 1. Evaluate Pretrained (with optional interventions)
    if args.pretrained_model:
        # Resolve model name
        pretrained_path, pretrained_name = _resolve_model_name(args.pretrained_model)
        model_name_for_summary = pretrained_name
        
        variant_name = "pretrained_base"
        if args.interventions_path:
            variant_name = "pretrained_with_interventions"
            
        results.append(
            _eval_variant(
                variant_name=variant_name,
                model_path=pretrained_path,
                model_name=pretrained_name,
                examples=examples,
                out_dir=args.out_dir,
                dtype=dtype,
                device_map=args.device_map,
                local_files_only=args.local_files_only,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
                interventions_path=args.interventions_path,
                heads_path=args.heads_path,
                num_heads=args.num_heads,
                alpha=args.alpha,
                edit_mode=args.edit_mode,
                prompt_encoding_text=args.prompt_encoding_text,
            )
        )

    # 2. Evaluate Finetuned (if provided)
    if args.finetuned_model is not None:
        # Resolve model name
        finetuned_path, finetuned_name = _resolve_model_name(args.finetuned_model)
        if model_name_for_summary is None:
            model_name_for_summary = finetuned_name
        
        results.append(
            _eval_variant(
                variant_name="finetuned",
                model_path=finetuned_path,
                model_name=finetuned_name,
                examples=examples,
                out_dir=args.out_dir,
                dtype=dtype,
                device_map=args.device_map,
                local_files_only=args.local_files_only,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
                interventions_path=None,
                heads_path=None,
                num_heads=None,
                alpha=args.alpha,
                edit_mode=args.edit_mode,
                prompt_encoding_text=args.prompt_encoding_text,
            )
        )

    # Summary file with model name (and alpha if interventions used)
    if model_name_for_summary is None:
        model_name_for_summary = "unknown"
    if args.interventions_path is not None:
        alpha_str = str(args.alpha).replace('.', '_')
        summary_path = os.path.join(args.out_dir, f"summary_{model_name_for_summary}_alpha{alpha_str}.json")
    else:
        summary_path = os.path.join(args.out_dir, f"summary_{model_name_for_summary}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    print(json.dumps({"summary_path": summary_path, "results": results}, indent=2))


if __name__ == "__main__":
    main()

"""
Example usage:

# With interventions

python eval_gsm8k.py --pretrained_model llama3_8B --max_examples 500
python eval_gsm8k.py --pretrained_model mistral_7B --max_examples 500
python eval_gsm8k.py --pretrained_model vicuna_7B --max_examples 500

python eval_gsm8k.py \
    --pretrained_model /work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_72_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5 \
    --max_examples 500 \
    --interventions_path /work/hdd/bcxt/yian3/toxic/features/llama3_8B_toxigen_vicuna_center_direction.npy \
    --heads_path /work/hdd/bcxt/yian3/toxic/features/heads/False_llama3_8B_toxigen_vicuna_seed_2_top_72_heads_fold_0.npy \
    --num_heads 18 \             
    """