# SFT Data Guide — from evidence files to a fine-tuned SLM

This answers two direct questions: **"how do I make paired instruction data?"** and **"how do I actually get an SFT job to train on it?"**

---

## 1. What "paired instruction data" means

A base language model just predicts the next token. Fine-tuning teaches it a *behavior* — answer this way when asked that — by showing it many examples of exactly that shape. Each example is a **pair**: something asked, and the answer you want the model to learn to produce. That's it. "Paired" just means every training row has both halves glued together; the model never sees a question without its matching answer during training.

Every file this project now produces (`sft_train.jsonl`, `sft_train_v2.jsonl`, `sft_train_v3.jsonl`) is already in paired form. One line = one JSON object = one (question, answer) pair, plus a system instruction that sets the behavior once for all of them.

---

## 2. The three JSONL shapes you'll run into

Different training tools expect slightly different field names for the same idea. Here is the same pair in all three, so you can convert freely:

**A. OpenAI / chat format** (what `build_sft_dataset*.py` already outputs — also read natively by HuggingFace `trl`'s `SFTTrainer`, Axolotl, and LLaMA-Factory with a one-line config change):

```json
{"messages": [
  {"role": "system", "content": "You are..."},
  {"role": "user", "content": "Which columns violate the prediction-time contract?"},
  {"role": "assistant", "content": "Observed evidence:\n- ..."}
]}
```

**B. Alpaca / instruction format** (older but still widely supported, e.g. by LLaMA-Factory, Axolotl):

```json
{"instruction": "Which columns violate the prediction-time contract?", "input": "", "output": "Observed evidence:\n- ..."}
```

**C. ShareGPT / conversations format** (used by some Axolotl and LLaMA-Factory configs, and by many open datasets on HuggingFace Hub):

```json
{"conversations": [
  {"from": "system", "value": "You are..."},
  {"from": "human", "value": "Which columns violate the prediction-time contract?"},
  {"from": "gpt", "value": "Observed evidence:\n- ..."}
]}
```

A 10-line converter between these is included at the bottom of this doc (`convert_format.py`) — you will not need to regenerate the underlying data to switch formats.

---

## 3. Quality bar before any of this touches a training job

The scripts already do the mechanical parts (dedup, stripping provenance noise, train/val split). Before you actually spend GPU time, check:

- **Length distribution.** Print token counts per example; drop or truncate extreme outliers (a 40-line holdout-metrics dump can dominate a batch).
- **Category balance.** Use the `metadata.source` / `metadata.stage` fields already in every example to check you're not 90% `optimization_reliability` and 2% `leakage_audit` — rebalance by upsampling the thin categories, not by inventing new ones.
- **A held-out slice the model never trains on**, separate from the mechanical val split — ask it the same style of question by hand afterward and read the answer, don't just trust the loss curve.
- **Mix in general instruction data (10–30%)** if the base model is small (1–3B parameters). A corpus this specialized, on its own, risks narrowing the model's general reasoning and writing ability — a well-known small-model fine-tuning failure mode. A few thousand examples from an open general instruction set (e.g. `HuggingFaceH4/no_robots` or `tatsu-lab/alpaca`) mixed in is enough for most cases.

---

## 4. An actual, runnable fine-tuning script

This uses HuggingFace `trl`'s `SFTTrainer` with LoRA (parameter-efficient — trains in a few hours on a single consumer/prosumer GPU, e.g. a 24GB card) on a small open instruct model. It reads the `messages`-format JSONL directly — no conversion needed for the output of `build_sft_dataset*.py`.

```python
# train_sft.py
# pip install -U transformers trl peft accelerate bitsandbytes datasets

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # or "meta-llama/Llama-3.2-3B-Instruct"

# 1. Load your paired data — concatenate v1 + v3 (and mixed-in general data) beforehand
#    into one train file and one val file if you want a single run.
dataset = load_dataset(
    "json",
    data_files={"train": "sft_train_v3.jsonl", "validation": "sft_val_v3.jsonl"},
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

# 4-bit quantization keeps this runnable on a single 24GB GPU
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

sft_config = SFTConfig(
    output_dir="dclab-slm-checkpoint",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="epoch",
    bf16=True,
    max_length=2048,          # DCLab's evidence-grounded examples run long — don't cut this too short
    packing=False,             # keep pairs unpacked so system/user/assistant boundaries stay clean
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=lora_config,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model("dclab-slm-final")
```

Run it:

```bash
python train_sft.py
```

`SFTTrainer` auto-detects the `messages` column and applies the model's own chat template — this is why the `messages` format from `build_sft_dataset*.py` needs no conversion.

For the OpenAI hosted fine-tuning API instead of a local GPU, the same `messages`-format JSONL uploads directly:

```bash
openai api fine_tuning.jobs.create -t sft_train_v3.jsonl -m gpt-4o-mini-2024-07-18
```

---

## 5. `convert_format.py` — if a tool needs Alpaca or ShareGPT instead

```python
import json, sys

def to_alpaca(ex):
    msgs = ex["messages"]
    user = next(m["content"] for m in msgs if m["role"] == "user")
    assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
    return {"instruction": user, "input": "", "output": assistant}

def to_sharegpt(ex):
    role_map = {"system": "system", "user": "human", "assistant": "gpt"}
    return {"conversations": [{"from": role_map[m["role"]], "value": m["content"]} for m in ex["messages"]]}

if __name__ == "__main__":
    in_path, out_path, fmt = sys.argv[1], sys.argv[2], sys.argv[3]  # fmt: alpaca | sharegpt
    convert = {"alpaca": to_alpaca, "sharegpt": to_sharegpt}[fmt]
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            fout.write(json.dumps(convert(json.loads(line)), ensure_ascii=False) + "\n")
```

```bash
python convert_format.py sft_train_v3.jsonl sft_train_v3_alpaca.jsonl alpaca
```
