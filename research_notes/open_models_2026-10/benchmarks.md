# Open-weight models for DCLab: public benchmark evidence (snapshot 2026-10-23)

Scope: models that fit, or nearly fit, a 32–64 GB Mac (≤ ~35B total, or a MoE with few active parameters),
plus 1–4B models. DCLab needs three abilities: (1) a multi-turn tool-calling ML agent that never leaks,
(2) a step-by-step ML-workflow reasoner that writes code, (3) small fast decision models.

How to read this file
- **[I]** = independent measurement (third party ran the model). **[V]** = vendor self-reported.
  **[V-other]** = one vendor measuring a competitor in its own table (still not neutral).
- Every number names its source and date. "no data" = I searched and found nothing trustworthy.
- Raw snapshots saved beside this file:
  - `aa_snapshot_2026-10-23.tsv`: Artificial Analysis per-model evaluation scores, read from the data embedded
    in artificialanalysis.ai model pages on 2026-10-23 (index v4.3.2). Scores ×100.
  - `bfcl_v4_overall_lastupdated_2026-04-12.csv`: the official BFCL v4 table (gorilla.cs.berkeley.edu, "Last Updated 2026-04-12").

---

## 1. The short answer (critical)

1. **No public benchmark measures what DCLab most needs: leakage-safe ML work.** MLE-bench rewards medals,
   not honest validation. SWE-bench and Terminal-Bench reward passing tests. A model can score high and still
   use the target's future. DCLab must keep its own eval (the planted-trap suites it already has) as the judge.
2. **At this size, the Qwen 27B dense line leads almost every independent agent number.** Qwen3.6-27B (Apr 2026)
   and Qwen3.8-27B (Aug 2026, Apache-2.0) beat Gemma 4, Nemotron 3.5 Lightning, gpt-oss-20b, Devstral Small 2,
   Mistral, Granite and OLMo on Artificial Analysis's terminal, long-context and science-code tests.
3. **Vendor numbers are inflated or not comparable.** The same model gets very different scores in different harnesses
   (examples in §6). Fresh, contamination-free tests are much lower: Qwen3.6-27B is 77.2 on SWE-bench Verified [V]
   but 31.2 on SWE-rebench (May–Jul 2026 tasks) [I].
4. **Several key leaderboards are stale.** BFCL (last update 2026-04-12), SWE-bench (last entry 2026-02-26),
   Aider polyglot (2025-11-20) and MLE-bench (closed to submissions 2026-04-24) have **no** entries for
   Qwen3.5/3.6/3.8, Gemma 4 or Nemotron 3.5.
5. **Fine-tuning gives large gains on narrow agent/data tasks** (often 2–10×), but gains are measured on the
   training-like benchmark; forgetting is rarely measured. Plan to measure it yourself.
6. **Persian: no current-generation benchmark data.** MMLU-ProX does not include Persian. Build a small Persian eval.

---

## 2. Candidates, size, licence, Mac fit

Memory = rough 4-bit weight size (≈0.55–0.6 GB per billion parameters), before the KV cache. My estimate, not measured.

| Model (release) | Total / active | Licence | ~4-bit weights | 32 GB Mac | 64 GB Mac |
|---|---|---|---|---|---|
| Qwen3.8-27B (2026-08-14) | 27B dense (+vision) | Apache-2.0 | ~16 GB | yes, short context | yes |
| Qwen3.6-27B (2026-04-22) | 27.8B dense | Apache-2.0 | ~16 GB | yes, short context | yes |
| Qwen3.6-35B-A3B (2026-04-16) | 36B / 3B | Apache-2.0 | ~20 GB | tight | yes, fast |
| Qwen3.5-9B / 4B / 2B / 0.8B (2026-03-02) | dense | Apache-2.0 | 5 / 2.7 / 1.3 / 0.5 GB | yes | yes |
| Gemma 4 31B / 26B-A4B (2026-04-02) | 30.7B dense / 25.2B-3.8B | Apache-2.0 | ~18 / ~15 GB | tight / yes | yes |
| Gemma 4 E4B / E2B (2026-04) | 8B (4.5B eff.) / 5.1B (2.3B eff.) | Apache-2.0 | ~4.5 / ~3 GB | yes | yes |
| Nemotron 3.5 Lightning (2026-08-11) | 31.6B / 3.6B | OpenMDW-1.1; data + recipes open | ~18 GB | tight | yes |
| GLM-4.7-Flash (2026-01) | 31.2B / 3B | MIT | ~18 GB | tight | yes |
| gpt-oss-20b (2025-08-05) | 21B / 3.6B | Apache-2.0 | ~13 GB (MXFP4 native) | yes | yes |
| Devstral Small 2 (2025-12) | 24B dense | Apache-2.0 | ~14 GB | yes | yes |
| Muse Glimmer 30B (Meta, 2026-08-10) | 30B | Apache-2.0 | <20 GB (Meta's 4-bit) | tight | yes |
| K2 Horizon 36B-A4B / 7B / 3.7B (MBZUAI IFM, 2026-09-03) | 36B-4B / 7B / 3.7B | Apache-2.0, data open | ~20 / 4 / 2 GB | – | yes |
| Mistral Small 4 (2026-03-16) | 119B / 6.5B | Apache-2.0 | ~65 GB | no | **no** (needs ~3-bit) |
| Granite 4.1/4.2 (30B, 8B, 3B) | dense | Apache-2.0 | – | yes | yes |
| OLMo 3 / 3.1 (7B, 32B) | dense | Apache-2.0, data open | – | yes | yes |
| Out of range: DeepSeek V4 Flash 284B-13B, V4.1 Flash 552B-16B; GLM-5.x (744B+; GLM-5.3-Flash 320B-18B); MiniMax M2.x/M3 (230B+); Kimi K3; Qwen3.8-Flash-Next (180B-6B) | | | | | |

Notes: Qwen3.8 has no open model below 27B (systemdesign.academy, Aug 2026; Qwen HF card). No Phi-5, Hermes 5,
Apriel 2 or DeepSeek V4 small distill was found (searches 2026-10-23); latest are Phi-4-reasoning-vision-15B
(Mar 2026), Hermes 4 (Aug 2025), Apriel-v1.6-15B (Nov 2025).

---

## 3. Independent scores: Artificial Analysis (read 2026-10-23, index v4.3.2) [I]

τ² = τ²-bench **telecom** only. TB-Hard = Terminal-Bench Hard; TB2.1 = Terminal-Bench 2.1 (AA's own harness).
LCR = AA-LCR long-context reasoning (10k–100k tokens). Halluc. = AA-Omniscience hallucination rate (lower is better).
"–" = AA did not run it (AA stopped running τ² and TB-Hard for models added after ~mid-2026).

| Model (reasoning mode unless noted) | AA index | τ² telecom | TB-Hard | TB2.1 | SciCode | LCR | IFBench | GPQA | HLE | Halluc. |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B (xhigh) | 33.7 | – | – | 79.8 | 46.6 | 82.0 | – | 90.5 | 33.9 | 30.3 |
| Qwen3.8-27B (non-reasoning) | 20.2 | – | – | 49.1 | 36.2 | 69.3 | – | 81.8 | 12.1 | 18.1 |
| Qwen3.6-27B | 21.4 | 94.2 | 34.8 | 60.7 | 42.8 | 77.3 | 67.6 | 84.2 | 23.1 | 49.3 |
| Qwen3.6-35B-A3B | 18.2 | 95.3 | 34.8 | 44.9 | 36.6 | 71.7 | 64.4 | 84.1 | 22.2 | 50.5 |
| Qwen3.5-27B | 22.9 | 93.9 | 32.6 | – | – | 77.7 | 75.6 | 85.8 | 23.9 | 81.5 |
| K2 Horizon MoVA 36B-A4B | 25.3 | – | – | 56.6 | 40.0 | 71.7 | – | 82.2 | 23.4 | 30.5 |
| Muse Glimmer 30B (high) | 17.5 | – | – | 51.7 | 44.9 | 83.3 | – | 83.5 | 22.0 | 81.9 |
| Gemma 4 31B | 14.7 | 59.9 | 36.4 | 43.4 | 45.5 | 69.7 | 75.6 | 85.7 | 23.6 | 85.0 |
| Gemma 4 26B-A4B | 16.7 | 43.6 | 13.6 | 39.0 | 40.0 | 65.7 | 72.4 | 79.2 | 19.3 | 86.4 |
| GLM-4.7-Flash | 14.9 | **98.8** | 22.0 | – | – | 41.7 | 60.8 | 58.1 | 7.6 | 93.9 |
| Nemotron 3.5 Lightning | 12.9 | – | – | 24.3 | 32.1 | 60.3 | – | 74.3 | 10.6 | 37.6 |
| Nemotron 3 Nano 30B-A3B | 8.9 | 40.9 | 13.6 | 6.7 | 30.6 | 38.0 | 71.1 | 75.7 | 11.4 | 83.3 |
| gpt-oss-20b (high) | 9.0 | 60.2 | 10.6 | 13.9 | 38.9 | 34.7 | 65.1 | 68.8 | 11.0 | 94.1 |
| Devstral Small 2 (24B) | 7.5 | 23.4 | 16.7 | 29.6 | 32.4 | 28.0 | 31.2 | 53.2 | 3.5 | 87.3 |
| Magistral Small 2509 (24B) | 8.6 | 27.8 | 4.5 | 4.5 | – | 19.3 | 44.4 | 66.3 | 6.4 | 90.8 |
| Mistral Small 4 (119B-6.5B) | 11.3 | 41.2 | 17.4 | 21.0 | 38.8 | 49.7 | 48.2 | 76.9 | 9.9 | 66.5 |
| Ministral 3 14B | 6.0 | 27.2 | 4.5 | 9.7 | 23.8 | 26.3 | 32.0 | 57.2 | 4.6 | 92.5 |
| Apriel v1.6 15B Thinker | 13.4 | 69.3 | 14.4 | – | – | – | 69.1 | 73.3 | 10.8 | 91.9 |
| EXAONE 4.5 33B | 13.2 | 78.1 | 20.5 | 21.3 | – | 54.7 | 58.0 | 79.4 | 12.9 | 79.9 |
| Granite 4.2 30B | 12.7 | – | – | 26.6 | 37.8 | 49.0 | – | 64.4 | 11.2 | 25.6 |
| Granite 4.1 30B | 7.4 | 42.1 | 2.3 | 2.6 | – | 23.7 | 44.4 | 48.1 | 4.1 | 95.0 |
| OLMo 3.1 32B Think | 7.1 | 0.0 | 0.0 | – | – | 0.0 | 66.0 | 59.1 | 6.3 | 66.3 |
| Seed-OSS 36B | 12.1 | 49.4 | 6.8 | – | – | 61.3 | 41.9 | 72.6 | 9.9 | 86.4 |
| Llama 4 Scout (109B-17B) | 6.5 | 15.5 | 1.5 | 3.7 | 21.3 | 27.7 | 39.5 | 58.7 | 3.8 | 79.4 |

Caveats for this table
- AA's index version changes over time. At launch AA wrote Muse Glimmer = 35 and Qwen3.6-27B = 38 (AA article, 2026-08-10);
  under v4.3.2 they are 17.5 and 21.4. Compare models only inside one snapshot.
- Qwen3.8-27B "xhigh" uses many tokens: 200M tokens to run the AA index (AA model page), so it is slow and costly per task.
- One run per model; AA publishes no confidence intervals in these fields.
- τ² telecom is near ceiling for the top models (GLM-4.7-Flash 98.8), so it no longer separates them.

---

## 4. Vendor-reported agentic and coding numbers [V]

| Model | SWE-bench Verified | Terminal-Bench | τ²/τ³ | BFCL-V4 | LiveCodeBench v6 | Source (date) |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | – (SWE-Pro 61.7) | TB2.1 73.0 | – | – | 90.3 | HF card, Aug 2026 |
| Qwen3.6-27B | 77.2 | TB2.0 59.3 | – | – | 83.9 | HF card, Apr 2026 |
| Qwen3.6-35B-A3B | 73.4 | TB2.0 51.5 | TAU3 67.2 | – | 80.4 | HF card, Apr 2026 |
| Qwen3.5-9B | – | – | TAU2 79.1 | 66.1 | 65.6 | Qwen3.5-4B HF card, Feb/Mar 2026 |
| Qwen3.5-4B | – | – | TAU2 79.9 | 50.3 | 55.8 | same |
| Gemma 4 31B | – | – | τ² avg-3 76.9 (retail 86.4) | – | 80.0 | HF card / DeepMind page, Apr 2026 |
| Gemma 4 26B-A4B | – | – | 68.2 | – | 77.1 | same |
| Gemma 4 E4B / E2B | – | – | 42.2 / 24.5 | – | 52.0 / 44.0 | same |
| Nemotron 3.5 Lightning | 51.56 | TB2.1 24.58 | τ³ banking 9.28 | – | – | NVIDIA HF card, 2026-08-11 |
| GLM-4.7-Flash | 59.2 | – | 79.5 | – | 64.0 | Z.ai HF card, Jan 2026 |
| gpt-oss-20b (high) | 60.7 | – | τ-bench retail 54.8 | – | – | OpenAI model card arXiv 2508.10925, Aug 2025 |
| Devstral Small 2 | 68.0 | TB2 22.5 | – | – | – | Mistral HF card, Dec 2025 |
| Granite 4.1 30B / 8B / 3B | – | – | – | BFCL **v3** 73.7 / 68.3 / 60.8 | – | IBM (via aimadetools blog), 2026 |

Gemma 4 long context [V]: MRCR v2 8-needle 128k: 31B 66.4, 26B-A4B 44.1, E4B 25.4, E2B 19.1 (HF card).
Qwen3.5 small long context [V]: AA-LCR 9B 63.0 / 4B 57.0; LongBench v2 9B 55.2 / 4B 50.0 (HF card).

---

## 5. Leaderboard status (what independent boards exist and how fresh)

| Board | Last update | Current candidates on it? | Best relevant facts |
|---|---|---|---|
| BFCL v4 (Berkeley) | 2026-04-12 | **No** Qwen3.5+, Gemma 4, gpt-oss, Nemotron 3.x | Older gen only: Qwen3-32B 48.7 overall; Qwen3-4B-2507 35.7; Nanbeige4-3B-Thinking 51.4; xLAM-2-8b multi-turn 70.0 |
| SWE-bench Verified (swebench.com) | last entry 2026-02-26 | No | Devstral Small 2512 56.4 with mini-SWE-agent [I] (vendor says 68.0) |
| SWE-rebench (Nebius), tasks 2026-05-15→07-01 | Oct 2026 view | Qwen only | Qwen3.6-27B 31.2 (pass@5 57.7); Qwen3.6-35B-A3B 24.7; Qwen3.5-35B-A3B 17.1 [I] |
| Terminal-Bench (tbench.ai) | now shows TB 4.0 only | – | Use AA's TB2.1 column instead |
| MLE-bench (OpenAI GitHub) | closed to submissions 2026-04-24 | **No** ≤35B open model | Best open: ML-Master 2.0 + DeepSeek-V3.2-Speciale 56.44% any-medal (2025-12-16) |
| MLE-bench Lite (BenchLM, 2026-10-19) | self-reports only | – | "Agents-A1-4B 22.7%" (self-reported, unverified) |
| Aider polyglot | 2025-11-20 | No | Qwen3-32B 40.0% is the best ≤40B entry |
| τ²-bench official (taubench.com) | via summary 2026-10-17 | No small models | Only 8 entries on the Overall board |
| DABstep (Adyen) | 2026 | Qwen3 only inside agent systems | Scores reflect the agent harness, not the bare model |

---

## 6. Harness variance: why vendor numbers mislead

| Model | Benchmark | Score A | Score B | Score C |
|---|---|---|---|---|
| Gemma 4 26B-A4B | SWE-bench Verified | 17.4 (Qwen's table [V-other], Apr 2026) | 57.40 (NVIDIA's table [V-other], Aug 2026) | – |
| Qwen3.6-35B-A3B | SWE-bench Verified | 73.4 (Qwen [V]) | 70.12 (NVIDIA [V-other]) | SWE-rebench 24.7 [I] |
| Qwen3.6-35B-A3B | Terminal-Bench | TB2.0 51.5 (Qwen [V]) | TB2.1 44.38 (NVIDIA [V-other]) | TB2.1 44.9 (AA [I]) |
| Devstral Small 2 | SWE-bench Verified | 68.0 (Mistral [V]) | 56.4 (mini-SWE-agent [I]) | – |
| Gemma 4 31B | τ² | 76.9 avg / 86.4 retail [V] | 59.9 telecom (AA [I]) | – |
| Gemma 4 E4B | τ² | 42.2 [V] | 20.8 telecom (AA [I]) | – |

Lesson: compare models only in one harness, and prefer fresh tasks (SWE-rebench) over static sets.

---

## 7. Data-science / tabular evidence

No public result for the current candidates on DSBench, DA-Code, InfiAgent-DABench or MLE-bench. What exists:

- **DARE-bench** (arXiv 2602.24288, ICLR 2026; 6,300 Kaggle-derived tasks; 5 turns, 200 s sandbox): base Qwen3-32B total 23.25,
  Qwen3-4B 4.39; gpt-4.1 ≈ 52–58 on modelling. **SFT Qwen3-32B → 41–43; RL (GRPO) Qwen3-4B → 37.40.** Time-series
  counterfactual stays ≈0 for all. GRPO's group normalisation made RL unstable; they removed it (Dr. GRPO style).
- **DataMind** (arXiv 2509.25084, ICLR 2026; DABench, TableBench, BIRD): Qwen2.5-Coder-7B 11.26 → DataMind-7B 68.10;
  Qwen2.5-Coder-14B 56.64 → DataMind-14B 71.16 (GPT-5 69.44, DeepSeek-V3.1 70.58). Pure RL collapsed; too much SFT
  caused entropy collapse; a decaying SFT weight was needed. "Void turns" masked from the loss.
- **AgenticDataBench** (arXiv 2607.01647, Jul 2026): only Qwen3.5-397B-A17B, Kimi-K2.5, Claude Sonnet 4.6;
  totals 40–49 for all; the harness changes the ranking (Claude 46.6 in Claude Code but 31.6 in CodeX).
- **LongDS-Bench** (arXiv 2605.30434, EMNLP 2026 sub.): 68 Kaggle-notebook tasks over 2,225 turns; best of five models 48.45%;
  ~47-point drop from early to late turns; long-horizon errors are 52–69% of failures. Directly relevant to DCLab's long sessions.
- **Nemotron-Terminal** (arXiv 2602.21193): on TB2.0 categories "Data Querying" and "Model Training", base Qwen3-14B/32B
  scored 0.0; Nemotron-Terminal-32B scored 60.0 and 50.0 after SFT.

---

## 8. Fine-tuning gains and risks

Gains (all measured on the benchmark the training targeted):
| Base → tuned | Benchmark | Before → after | Source |
|---|---|---|---|
| Qwen3-8B → Nemotron-Terminal-8B (SFT) | Terminal-Bench 2.0 | 2.47 → 13.0 (community repro 15.9) | arXiv 2602.21193 |
| Qwen3-14B → Nemotron-Terminal-14B | TB2.0 | ~4.0 → 20.2 | same (+ featherless page) |
| Qwen3-32B → Nemotron-Terminal-32B | TB2.0 | 3.37 → 27.4 | same |
| Qwen3-8B → OpenThinker-Agent-v1 (SFT+RL) | TB2.0 | 0.0 → 4.9 (hosting-page figure, unverified) | featherless.ai |
| Llama-3.1-8B-Instruct → xLAM-2-8b | BFCL v4 multi-turn | 11.1 → 70.0 | BFCL CSV 2026-04-12 [I] |
| same | BFCL v4 web search / memory | 3.0 → 6.5 / 10.8 → 14.0 | same: the narrow skill rose, the rest barely moved |
| Qwen3-32B (SFT) / Qwen3-4B (RL) | DARE-bench total | 23.25 → ~42.9 / 4.39 → 37.40 | arXiv 2602.24288 |
| Qwen2.5-Coder-7B → DataMind-7B | DA benchmarks avg | 11.26 → 68.10 | arXiv 2509.25084 |

Risks and instability reports:
- **Forgetting:** QLoRA tool-knowledge study (arXiv 2605.17774, May 2026): Qwen3-4B learned well but forgot more on
  general multiple-choice than Gemma; LoRA rank trades task quality against retention. DARE-bench and DataMind did not measure forgetting.
- **Specialisation trade-off:** xLAM-2 models top BFCL multi-turn but are weak on web-search/memory parts (table above).
- **RL instability:** DataMind (pure RL collapses), DARE-bench (GRPO normalisation), arXiv 2606.26027 (Qwen3: SFT data without
  reasoning wrappers + RL with thinking → collapse).
- **MoE fine-tuning:** Unsloth's Qwen3.5 guide: router-layer training disabled by default for stability; 4-bit QLoRA for MoE
  not recommended (bitsandbytes). So 35B-A3B models need bf16 LoRA (~70 GB of weights): **not on a 64 GB Mac**.
- **Gemma 4 E2B/E4B:** layers share KV state; with `use_cache=False` (common in QLoRA recipes) attention can be corrupted
  (Transformers issue #45242, per Unsloth's Gemma 4 guide). Expected loss on E2B/E4B is 13–15 (normal, per Unsloth).
- **Qwen3.5/3.6 templates:** tool calls use the Qwen3-Coder XML format; wrong parsers break tool calling (Ollama issue #14493);
  Qwen3.6 drops earlier-turn reasoning unless `preserve_thinking` is set, which caused empty tool arguments (pi issue #3325).
  Fine-tune with the official template, or the model learns a format the runtime will not parse.
- **Mac training:** I found no confirmation that mlx-lm LoRA supports Qwen3.5/3.6's Gated DeltaNet layers above 0.8B. Unverified.

---

## 9. Small models (1–4B) for tool calls and decisions

| Model | τ² telecom [I] AA | TB2.1 [I] | LCR [I] | IFBench [I] | Vendor |
|---|---|---|---|---|---|
| Qwen3.5-4B (reasoning) | 92.1 | 25.8 | 63.0 | 52.0 | TAU2 79.9, BFCL-V4 50.3 |
| Qwen3.5-4B (non-reasoning) | 87.7 | 21.3 | 34.7 | 33.3 | – |
| Qwen3.5-2B (non-reasoning) | 81.6 | 0.0 | 14.0 | 29.1 | – |
| K2 Horizon 3.7B | – | 28.1 | 62.3 | – | – |
| MiniCPM5-2B (2026-09-07) | – | 8.6 | 59.3 | – | – |
| MiniCPM5-1B | 81.0 | – | 6.0 | 49.3 | – |
| Gemma 4 E4B (reasoning) | 20.8 | 1.9 | 32.0 | 44.2 | τ² 42.2 |
| Gemma 4 E2B | 20.8 | 0.4 | 16.3 | 38.0 | τ² 24.5 |
| Nemotron 3 Nano 4B | 28.1 | 3.7 | 21.7 | 58.2 | – |
| Granite 4.2 3B | – | 13.9 | 24.3 | – | – |
| Ministral 3 3B | 24.9 | 0.0 | 17.0 | 26.8 | – |
| Phi-4-mini (3.8B) | 8.2 | 0.4 | 15.3 | 21.1 | – |
| Nanbeige4.1-3B | 21.6 | 1.1 | 0.0 | 35.4 | – |

BFCL v4 (older generation, 2026-04-12) [I]: Nanbeige4-3B-Thinking 51.4 overall; xLAM-2-3b 41.2 (multi-turn 58.4);
Qwen3-4B-Instruct-2507 35.7; Arch-Agent-3B 35.4; Qwen3-1.7B 28.4; Granite-4.0-350m 19.0; Gemma-3-4b 19.6.

Notes
- Qwen3.5-4B is the clear small leader in independent data. Qwen3.5-2B is usable only for short single-step calls.
- An Ertas blog (2026-05/08) put Qwen3-4B-2507 in the "high 80s" on BFCL v4; the official table says 35.7. Ignore such blogs.
- For classification-style decisions, no study compares Qwen3.5 0.8–4B against encoders (ModernBERT, DeBERTa, GLiNER) on
  tabular-column decisions. Older studies (arXiv 2406.08660, 2508.18253) show fine-tuned encoders match or beat fine-tuned
  small LLMs on classification. DCLab already has classifiers per decision point; an LLM must beat them on DCLab's sealed sets.

---

## 10. Persian (Farsi)

- **No current-generation data** (Qwen3.6/3.8, Gemma 4, Nemotron 3.5) found on any Persian benchmark (searches 2026-10-23).
- MMLU-ProX (29 languages) does **not** include Persian (mmluprox.github.io). So Qwen's MMLU-ProX numbers say nothing about Farsi.
- DivanBench (arXiv 2602.17623, EACL 2026 workshop): Gemma 3 12B best of seven 7–12B models (87% factual, 66% cultural);
  most models accept wrong statements ("acquiescence trap"). Source: Persian-AI blog (2026-08-09, upd. 2026-10-07), not checked in the paper.
- Same blog says Qwen3.5-9B led a 24-language summarisation study (arXiv 2606.01252); the abstract does not name Persian. Unverified.
- PARSA-Bench (arXiv 2603.14456): audio; Gemma E4B ~0.86 on Persian reading comprehension; Qwen3-30B strongest overall.
- Recommendation: build a 100–200-item Persian eval from DCLab's own explanations (stage reports, questions) and score candidates on it.

---

## 11. What this means for DCLab's three models (my judgment, not a benchmark)

1. **Agent (tool calls over many turns):** start from **Qwen3.6-27B or Qwen3.8-27B** (dense, Apache-2.0). They lead the
   independent terminal, long-context and science-code numbers at this size. Weak points: slow on a Mac (dense 27B),
   token-hungry in high reasoning mode, and only 31% on fresh SWE tasks. Train in the cloud (bf16 LoRA), serve 4-bit on the Mac.
2. **Fast agent / second opinion:** **Qwen3.6-35B-A3B** (3B active, fast) or **Nemotron 3.5 Lightning** (weaker, but training
   data and recipes are open, which helps honest contamination checks). GLM-4.7-Flash is strong on τ² but weak on long context (41.7).
3. **Small decision models:** **Qwen3.5-4B** first, Qwen3.5-2B for latency; K2 Horizon 3.7B as a fully-open check.
   Always compare against the existing classifiers; a 4B LLM is not automatically better on structured decisions.
4. **Gemma 4:** fine for multimodal and general text, but independent agent numbers are clearly below Qwen at every size.
5. **Skip for this job:** OLMo 3.x, Granite 4.1, Phi-4-mini, Ministral 3, Llama 4 (very low independent agent scores).

Open risks: harness dependence; contamination of static sets; no leakage-aware benchmark; forgetting not measured;
MoE fine-tuning does not fit a 64 GB Mac; Mac (MLX) training support for Gated DeltaNet unverified.

---

## Sources (accessed 2026-10-23)

- Qwen3.6-27B card: https://huggingface.co/Qwen/Qwen3.6-27B ; Qwen3.6-35B-A3B: https://huggingface.co/Qwen/Qwen3.6-35B-A3B ;
  Qwen3.8-27B: https://huggingface.co/Qwen/Qwen3.8-27B ; Qwen3.5-4B (small-model table): https://huggingface.co/Qwen/Qwen3.5-4B
- Gemma 4 card: https://huggingface.co/google/gemma-4-31B-it ; DeepMind page: https://www.deepmind.google/models/gemma/gemma-4
- Nemotron 3.5 Lightning card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 ;
  NVIDIA blog 2026-08-11: https://developer.nvidia.com/blog/nvidia-nemotron-3-5-lightning-delivers-fast-accurate-specialized-task-execution-for-long-running-agents/ ;
  AA launch article 2026-08-11: https://artificialanalysis.ai/articles/nemotron-3-5-lightning-launch
- GLM-4.7-Flash card: https://huggingface.co/zai-org/GLM-4.7-Flash ; Devstral Small 2 card: https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512
- gpt-oss model card: https://arxiv.org/pdf/2508.10925 ; Mistral Small 4: https://mistral.ai/en/news/mistral-small-4/
- Artificial Analysis model pages (embedded data): https://artificialanalysis.ai/models/qwen3-6-27b , https://artificialanalysis.ai/models/open-source ;
  Muse Glimmer article 2026-08-10: https://artificialanalysis.ai/articles/muse-glimmer
- Leaderboard audit (2026-10-19): https://dev.to/ward_ed_6b5e6aa8ded94a987/best-open-weight-llm-in-october-2026-auditing-two-leaderboards-row-by-row-1b64
- BFCL: https://gorilla.cs.berkeley.edu/leaderboard.html (data_overall.csv) ; SWE-bench data: https://github.com/SWE-bench/swe-bench.github.io (data/leaderboards.json) ;
  SWE-rebench: https://swe-rebench.com/ ; MLE-bench: https://github.com/openai/mle-bench ; Aider: https://aider.chat/docs/leaderboards/ ;
  BenchLM MLE-bench Lite: https://benchlm.ai/benchmarks/mle-bench-lite
- DARE-bench: https://arxiv.org/abs/2602.24288 ; DataMind: https://arxiv.org/abs/2509.25084 ; AgenticDataBench: https://arxiv.org/abs/2607.01647 ;
  LongDS-Bench: https://arxiv.org/abs/2605.30434 ; Nemotron-Terminal: https://arxiv.org/abs/2602.21193 ; QLoRA tool knowledge: https://arxiv.org/abs/2605.17774 ;
  tool-use RL collapse: https://arxiv.org/abs/2606.26027
- Unsloth guides: https://unsloth.ai/docs/models/gemma-4/train , https://unsloth.ai/docs/models/qwen3.6 ; Ollama issue: https://github.com/ollama/ollama/issues/14493
- Persian: https://hussain-nazary.github.io/persian-ai-2026-local-models.html ; https://mmluprox.github.io/ ; https://arxiv.org/abs/2603.14456
- K2 Horizon: https://aiweekly.co/alerts/mbzuais-ifm-ships-k2-horizon-six-fully-open-models-from-09b-to-375b-parameters
