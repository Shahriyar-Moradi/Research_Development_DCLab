"""
make_tiny_model.py
==================

A tiny, randomly initialised model with Qwen3.5's architecture (Gated DeltaNet + gated attention) and Qwen3.5's
real chat template, for checking the Mac pipeline end to end in minutes before spending hours on the 9B model:
traces, export, training (including resume) and evaluation all run on it. Its answers are noise; only the
mechanics are being tested.

    python mac/make_tiny_model.py --out models/tiny-qwen35
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build_tokenizer(out: Path, texts: list[str]):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
                "<tool_response>", "</tool_response>"]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=1500, special_tokens=specials,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    fast.chat_template = (HERE / "testdata" / "qwen35_chat_template.jinja").read_text().rstrip("\n")
    fast.save_pretrained(out)
    return len(fast)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, default=HERE.parent / "out" / "mix" / "items_train.jsonl")
    args = ap.parse_args()
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.utils import _get_classes

    args.out.mkdir(parents=True, exist_ok=True)
    texts = []
    if args.corpus.exists():
        for i, line in enumerate(open(args.corpus, encoding="utf-8")):
            if i > 400:
                break
            it = json.loads(line)
            texts += [m["content"] for m in it["context"]] + [it["answer"]]
    texts += ["DECISION: keep WHY: the evidence RULE: CONFIDENCE: high NEXT: continue", "ANSWER: 0.1234 def f(x): return x"]
    vocab = build_tokenizer(args.out, texts)
    config = {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
              "text_config": {"model_type": "qwen3_5_text", "hidden_size": 64, "intermediate_size": 128,
                              "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                              "linear_num_value_heads": 4, "linear_num_key_heads": 2, "linear_key_head_dim": 32,
                              "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
                              "vocab_size": vocab, "rms_norm_eps": 1e-6, "tie_word_embeddings": False,
                              "max_position_embeddings": 32768,
                              "rope_parameters": {"rope_type": "default", "mrope_section": [2, 1, 1], "rope_theta": 10000000,
                                                  "partial_rotary_factor": 0.25, "mrope_interleaved": True}}}
    model_class, args_class = _get_classes(config)
    model = model_class(args_class.from_dict(config))
    mx.eval(model.parameters())
    mx.save_safetensors(str(args.out / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (args.out / "config.json").write_text(json.dumps(config, indent=1))
    print(f"tiny Qwen3.5-architecture model written to {args.out} (vocabulary {vocab})")


if __name__ == "__main__":
    main()
