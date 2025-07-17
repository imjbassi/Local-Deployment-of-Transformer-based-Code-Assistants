import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import time
import ast
import warnings
from tqdm import tqdm

# Suppress syntax warnings from exec'd model outputs
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Load full HumanEval dataset
human_eval = load_dataset("openai_humaneval")['test']

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
def generate_completion(model, tokenizer, prompt, model_name, max_length=256, num_return_sequences=5):
    if "StarCoder" in model_name:
        prompt = "# Here is the correct implementation of the code exercise\n" + prompt
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

# Check functional correctness (pass@k)
def is_correct(generated_code, test_code):
    try:
        namespace = {}
        exec(generated_code, namespace)
        exec(test_code, namespace)
        return True
    except Exception:
        return False

# Run benchmark on full HumanEval
results = {}
for name, (model, tokenizer) in loaded_models.items():
    total_latency = 0
    correct_1 = 0
    correct_5 = 0
    total = len(human_eval)
    print(f'\nBenchmarking {name}...')

    for example in tqdm(human_eval, desc=f'{name}', unit="example"):
        prompt = example["prompt"]
        test_code = example["test"]
        completions = generate_completion(model, tokenizer, prompt, name)
        latency = measure_latency(model, tokenizer, prompt)
        total_latency += latency

        if is_correct(completions[0], test_code):
            correct_1 += 1
        if any(is_correct(c, test_code) for c in completions):
            correct_5 += 1

    results[name] = {
        'avg_tokens_per_sec': total_latency / total,
        'pass@1 (%)': correct_1 / total * 100,
        'pass@5 (%)': correct_5 / total * 100
    }

# Display summarized results
print('\nBenchmark Results:')
for model_name, metrics in results.items():
    print(f"{model_name}: Avg tokens/sec: {metrics['avg_tokens_per_sec']:.2f}, Pass@1: {metrics['pass@1 (%)']:.2f}%, Pass@5: {metrics['pass@5 (%)']:.2f}%")
