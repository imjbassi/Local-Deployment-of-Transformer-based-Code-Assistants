import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import time

# Load HumanEval dataset
human_eval = load_dataset("openai_humaneval")['test'].select(range(5))

# Models and Tokenizers
models = {
    'CodeT5-small': ('Salesforce/codet5-small', AutoModelForSeq2SeqLM),
    'CodeT5-base': ('Salesforce/codet5-base', AutoModelForSeq2SeqLM),
    'StarCoder-1B': ('bigcode/starcoderbase-1b', AutoModelForCausalLM)
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

loaded_models = {}
for name, (model_id, model_cls) in models.items():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = model_cls.from_pretrained(model_id).to(device)
    loaded_models[name] = (model, tokenizer)

# Benchmark generation
def generate_completion(model, tokenizer, prompt, max_length=256, num_return_sequences=5):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, 
                             max_new_tokens=max_length, 
                             num_return_sequences=num_return_sequences, 
                             do_sample=True, 
                             temperature=0.7,
                             top_p=0.95)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Latency measurement
def measure_latency(model, tokenizer, prompt, max_length=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    model.generate(**inputs, max_new_tokens=max_length)
    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.time()
    latency = max_length / (end - start)
    return latency

# Run benchmark on first 5 examples
results = {}
for name, (model, tokenizer) in loaded_models.items():
    total_latency = 0
    total = 0
    print(f'Benchmarking {name}:')
    for example in human_eval:
        prompt = example["prompt"]
        completions = generate_completion(model, tokenizer, prompt)
        latency = measure_latency(model, tokenizer, prompt)
        total_latency += latency
        print(f'Prompt: {prompt[:50]}...')
        print(f'Completion example: {completions[0][:50]}...')
        print(f'Latency: {latency:.2f} tokens/sec\n')
        total += 1

    avg_latency = total_latency / total
    results[name] = {'avg_tokens_per_sec': avg_latency}

# Display summarized results
print('\nBenchmark Results:')
for model_name, metrics in results.items():
    print(f"{model_name}: Avg tokens/sec: {metrics['avg_tokens_per_sec']:.2f}")
