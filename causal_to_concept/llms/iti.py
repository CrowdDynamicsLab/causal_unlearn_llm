import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from contextlib import contextmanager


# Load model and tokenizer
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Text generation function
def generate(prompt, max_new_tokens=40):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)



class ActivationIntervention:
    def __init__(self, layer_num=3, alpha=5.0):
        self.layer_num = layer_num
        self.alpha = alpha

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            output = output[0]  # unwrap the tuple
        
        direction = torch.randn_like(output)
        return output + self.alpha * direction

    @contextmanager
    def register(self, model):
        handle = model.transformer.h[self.layer_num].register_forward_hook(self)
        try:
            yield
        finally:
            handle.remove()


# Main script
if __name__ == "__main__":
    prompt = "The government of the United States"

    print("🧠 Original Output:")
    print(generate(prompt))

    print("\n🔧 With Inference-Time Intervention:")
    intervention = ActivationIntervention(layer_num=3, alpha=2.0)
    with intervention.register(model):
        print(generate(prompt))

    print("\n✅ Done.")
