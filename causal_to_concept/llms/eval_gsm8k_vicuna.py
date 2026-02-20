"""
Evaluate GSM8K performance.
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
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from einops import rearrange

try:
    from baukit import TraceDict
except Exception as e:  # pragma: no cover
    TraceDict = None
    _BAUKIT_IMPORT_ERROR = e


_FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?[\d,]+(?:\.\d+)?)")
_FALLBACK_NUMBER_RE = re.compile(r"([-+]?[\d,]+(?:\.\d+)?)")

# --- Standard 8-Shot CoT Examples ---
GSM8K_COT_EXAMPLES = """
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6. #### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. #### 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. #### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8. #### 8

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and his dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9. #### 9

Question: There were 9 computers in the server room. Five more computers were installed each day for 4 days. How many computers are now in the server room?
Answer: There were originally 9 computers. For 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29. #### 29

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33. #### 33

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is 8. #### 8
"""
# ------------------------------------------------------------------

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
    for i, row in enumerate(ds):
        examples.append(GSM8KExample(idx=i, question=row["question"], answer=row["answer"]))
    return examples


def _extract_final_number(text: str) -> Optional[str]:
    def clean(s): return s.replace(",", "").strip()
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        return clean(m.group(1))
    nums = _FALLBACK_NUMBER_RE.findall(text)
    if not nums:
        return None
    return clean(nums[-1])


def _resolve_model_name(model_input: str) -> Tuple[str, str]:
    if '/' in model_input or model_input.startswith('meta-llama/') or model_input.startswith('mistralai/'):
        clean_name = model_input.split('/')[-1].replace('-', '_').replace('.', '_')
        return model_input, clean_name
    if model_input in HF_NAMES:
        return HF_NAMES[model_input], model_input
    return model_input, model_input.replace('/', '_').replace('-', '_')


def _format_prompt(question: str, tokenizer: AutoTokenizer, is_instruct: bool, model_name_lower: str) -> str:
    """
    Handles prompt formatting for Base, Instruct, and specifically Vicuna.
    """
    cot_text = GSM8K_COT_EXAMPLES.strip()
    
    # 1. SPECIAL CASE: VICUNA
    # Older Vicuna models on HF often lack 'chat_template' in config, so we force it here.
    if "vicuna" in model_name_lower:
        # Standard Vicuna v1.0/v1.5 format
        system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        user_msg = f"Answer the following math problems step by step.\n\n{cot_text}\n\nQuestion: {question}"
        
        return (
            f"{system_prompt} "
            f"USER: {user_msg} "
            f"ASSISTANT:"
        )

    # 2. GENERIC INSTRUCT (Mistral, Llama-3-Instruct)
    if is_instruct and tokenizer.chat_template is not None:
        messages = [
            {"role": "user", "content": f"Answer the following math problems step by step.\n\n{cot_text}\n\nQuestion: {question}\nAnswer:"}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 3. BASE MODELS (Raw Completion)
    return (
        f"{cot_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _dtype_from_str(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _load_tokenizer(model_path: str, local_files_only: bool) -> Any:
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only, use_fast=True)
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token
    return tok


def _load_model(model_path: str, dtype: torch.dtype, device_map: str, local_files_only: bool) -> Any:
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        local_files_only=local_files_only,
    ).eval()


def _get_model_input_device(model: Any) -> torch.device:
    if hasattr(model, "device"):
        try:
            return model.device
        except Exception:
            pass
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def _tensorize_direction(direction: Any) -> torch.Tensor:
    if isinstance(direction, torch.Tensor):
        return direction
    return torch.tensor(direction, dtype=torch.float32)


def _load_interventions(path: str) -> Dict[str, List[Tuple[int, torch.Tensor, float]]]:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".pt", ".pth"}:
        raw = torch.load(path, map_location="cpu")
    elif ext in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            raw = pickle.load(f)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raise ValueError(f"Unsupported interventions file extension: {ext}")

    if not isinstance(raw, Mapping):
        raise TypeError("Interventions file must contain a dict-like mapping.")

    normalized: Dict[str, List[Tuple[int, torch.Tensor, float]]] = {}
    for layer_name, entries in raw.items():
        if entries is None:
            continue
        out_entries: List[Tuple[int, torch.Tensor, float]] = []
        entry_list = entries
        if isinstance(entries, dict):
             entry_list = []
             for h, d in entries.items():
                 entry_list.append((h, d, 1.0))

        for item in entry_list:
            if isinstance(item, dict):
                head = int(item["head"])
                direction = _tensorize_direction(item["direction"])
                proj_val_std = float(item.get("proj_val_std", 1.0))
            else:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                     continue
                head = int(item[0])
                direction = _tensorize_direction(item[1])
                proj_val_std = float(item[2]) if len(item) >= 3 and item[2] is not None else 1.0
            
            out_entries.append((head, direction, proj_val_std))
        normalized[str(layer_name)] = out_entries
    return normalized


def _load_heads_file(path: str, num_heads: int) -> List[Tuple[int, int]]:
    import numpy as np
    heads_array = np.load(path)
    if heads_array.ndim == 2 and heads_array.shape[1] == 2:
        return [(int(l), int(h)) for l, h in heads_array[:num_heads]]
    elif heads_array.ndim == 1:
        return [(int(h // 32), int(h % 32)) for h in heads_array[:num_heads]]
    raise ValueError(f"Unexpected heads array shape: {heads_array.shape}")


def _make_edit_output_fn(
    interventions: Mapping[str, Sequence[Tuple[int, torch.Tensor, float]]],
    num_heads: int,
    alpha: float,
    edit_mode: str,
) -> Any:
    if edit_mode not in {"last", "all"}:
        raise ValueError("--edit_mode must be 'last' or 'all'")

    def edit_output(output: torch.Tensor, layer_name: str) -> torch.Tensor:
        entries = interventions.get(layer_name)
        if not entries:
            return output
        if output.ndim != 3:
            return output

        bsz, seq_len, hidden = output.shape
        if num_heads <= 0 or hidden % num_heads != 0:
            return output
        head_dim = hidden // num_heads

        out_fp32 = output.detach().to(dtype=torch.float32)
        out_fp32 = out_fp32.view(bsz, seq_len, num_heads, head_dim)
        pos_slice = slice(seq_len - 1, seq_len) if edit_mode == "last" else slice(0, seq_len)

        for head, direction, proj_val_std in entries:
            if head < 0 or head >= num_heads:
                continue
            d = direction.to(device=out_fp32.device, dtype=out_fp32.dtype)
            d = d.view(1, 1, 1, head_dim)
            out_fp32[:, pos_slice, head, :] += (alpha * float(proj_val_std)) * d

        out_fp32 = out_fp32.view(bsz, seq_len, hidden)
        return out_fp32.to(dtype=output.dtype)

    return edit_output


@torch.inference_mode()
def _generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    interventions: Optional[Dict[str, List[Tuple[int, torch.Tensor, float]]]],
    edit_output_fn: Optional[Any],
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_device = _get_model_input_device(model)
    input_ids = inputs["input_ids"].to(input_device)
    attn_mask = inputs.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(input_device)

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    if interventions and edit_output_fn is not None:
        if TraceDict is None:
            raise RuntimeError("baukit TraceDict is not available.")
        layers_to_intervene = list(interventions.keys())
        with TraceDict(model, layers_to_intervene, edit_output=edit_output_fn):
            out = model.generate(input_ids=input_ids, attention_mask=attn_mask, **gen_kwargs)
    else:
        out = model.generate(input_ids=input_ids, attention_mask=attn_mask, **gen_kwargs)

    new_tokens = out[0, input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _eval_variant(
    *,
    variant_name: str,
    model_path: str,
    model_name: str,
    examples: Sequence[GSM8KExample],
    out_dir: str,
    dtype: torch.dtype,
    device_map: str,
    local_files_only: bool,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    interventions_path: Optional[str],
    heads_path: Optional[str],
    num_heads: Optional[int],
    alpha: float,
    edit_mode: str,
) -> Dict[str, Any]:
    
    tokenizer = _load_tokenizer(model_path, local_files_only=local_files_only)
    model = _load_model(model_path, dtype=dtype, device_map=device_map, local_files_only=local_files_only)

    # UPDATED: Vicuna detection is now explicit
    model_lower = model_path.lower()
    is_instruct = any(x in model_lower for x in ["instruct", "chat", "vicuna"])
    
    if is_instruct:
        print(f"Detected Instruct/Chat model: {model_path}")
        if "vicuna" in model_lower:
            print("Applying manual Vicuna template (USER/ASSISTANT).")

    interventions = None
    edit_output_fn = None
    
    if interventions_path is not None:
        all_interventions = _load_interventions(interventions_path)
        if heads_path is not None and num_heads is not None:
            top_heads = _load_heads_file(heads_path, num_heads)
            print(f"Filtering interventions to top {num_heads} heads")
            filtered_interventions = {}
            for layer_idx, head_idx in top_heads:
                layer_name = f"model.layers.{layer_idx}.self_attn.o_proj"
                source_entries = all_interventions.get(layer_name, [])
                for entry in source_entries:
                    if entry[0] == head_idx:
                        if layer_name not in filtered_interventions:
                            filtered_interventions[layer_name] = []
                        filtered_interventions[layer_name].append(entry)
                        break
            interventions = filtered_interventions
        else:
            interventions = all_interventions
            
        try:
            model_num_heads = int(model.config.num_attention_heads)
        except Exception as e:
            raise RuntimeError("Could not determine num_attention_heads") from e
        
        edit_output_fn = _make_edit_output_fn(interventions, num_heads=model_num_heads, alpha=alpha, edit_mode=edit_mode)

    _ensure_dir(out_dir)
    suffix = f"_alpha{alpha}".replace(".", "_") if interventions else ""
    pred_path = os.path.join(out_dir, f"{model_name}_{variant_name}{suffix}.jsonl")

    correct = 0
    total = 0
    with open(pred_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(tqdm(examples, desc=f"{model_name} {variant_name}")):
            # UPDATED: Pass model_name_lower to _format_prompt
            prompt = _format_prompt(ex.question, tokenizer, is_instruct, model_lower)
            
            completion = _generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                interventions=interventions,
                edit_output_fn=edit_output_fn,
            )
            pred = _extract_final_number(completion)
            gold = _extract_final_number(ex.answer)
            is_correct = pred is not None and gold is not None and abs(float(pred) - float(gold)) < 1e-6
            
            correct += int(is_correct)
            total += 1

            f.write(
                json.dumps(
                    {
                        "idx": ex.idx,
                        "question": ex.question,
                        "gold": gold,
                        "pred": pred,
                        "correct": bool(is_correct),
                        "full_completion": completion
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if (i + 1) % 5 == 0:
                f.flush()

    acc = correct / max(total, 1)
    print(f"Result for {variant_name}: {acc:.2%}")
    return {
        "variant": variant_name,
        "model": model_name,
        "accuracy": acc,
        "output_file": pred_path,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model", type=str, required=True)
    p.add_argument("--finetuned_model", type=str, default=None)
    p.add_argument("--interventions_path", type=str, default=None)
    p.add_argument("--heads_path", type=str, default=None)
    p.add_argument("--num_heads", type=int, default=None)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_examples", type=int, default=100)
    p.add_argument("--out_dir", type=str, default="results_gsm8k")
    p.add_argument("--dtype", type=str, default="float16")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--alpha", type=float, default=5.0)
    p.add_argument("--edit_mode", type=str, default="last")
    args = p.parse_args()

    dtype = _dtype_from_str(args.dtype)
    examples = _load_gsm8k(args.split, args.max_examples)
    _ensure_dir(args.out_dir)
    results = []
    
    _, model_name = _resolve_model_name(args.pretrained_model)

    if args.pretrained_model:
        v_name = "pretrained_base"
        if args.interventions_path:
            v_name = "pretrained_with_interventions"
        
        results.append(
            _eval_variant(
                variant_name=v_name,
                model_path=args.pretrained_model,
                model_name=model_name,
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
            )
        )

    if args.finetuned_model:
        _, ft_name = _resolve_model_name(args.finetuned_model)
        results.append(
            _eval_variant(
                variant_name="finetuned",
                model_path=args.finetuned_model,
                model_name=ft_name,
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
            )
        )

    summary_path = os.path.join(args.out_dir, f"summary_{model_name}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(json.dumps({"summary_path": summary_path, "results": results}, indent=2))

if __name__ == "__main__":
    main()

"""
python eval_gsm8k.py \
    --pretrained_model vicuna_7B \
    --max_examples 100 
"""