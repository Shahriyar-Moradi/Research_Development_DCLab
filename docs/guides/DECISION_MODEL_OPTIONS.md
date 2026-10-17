# Decision models beside DCLab's: three options compared (2026-10-17)

The owner asked which outside decision model DCLab should run beside its own rules and classifiers, and decided to
**collect, not train**: every answer such a model gives on DCLab's decisions is stored with every detail, as data for
training DCLab's own models later (package 26.9). Figures below are the vendors' own or third-party pages', not
DCLab's measurements; DCLab measures any of them on its own sealed decisions before trusting it.

| | Fastino | TypeSafe Jev | OpenAI Decisions API |
|---|---|---|---|
| Models | **GLiNER2.5-Decide** (340M encoder, DeBERTa-v3-large); **GLiDE** (hosted decision model, 131k context) | **Jev 1.13** ("System One": typed answers, choice / score / noul, with probabilities) | **gpt-6-luna** at `/v1/decisions` (predicate / choice / score), beta |
| Inputs | text | text (images announced) | text and images |
| Reported quality | GLiDE 64.8 on Fastino's Decision Index (Jev 57.9); GLiNER2.5 variants rank low on JevBench | JevBench 72.1 (#3), calibration 88, Brier 0.149 on typed decisions | JevBench 40.5 (#34), calibration 95, Brier 0.204 |
| Speed, price | GLiNER2.5: 44 ms on a GPU, 167 ms on a CPU, free locally (hosted about $0.03 per million tokens); GLiDE about $0.15-0.30 per million | about 135 ms; about $0.04 per million input tokens | about 110 ms; $0.10 per million input tokens |
| Fine-tuning, weights | GLiNER2.5-Decide: open weights, **Apache 2.0**, fine-tunable; GLiDE: no | closed; no | not offered |
| Its answers as training data | GLiNER2.5-Decide run locally: **yes** (no service terms); GLiDE: terms not found | terms not found (reported: Jev does not train on customer data) | **no**: the terms forbid using outputs to develop competing models |

**What DCLab does with them (26.9):** GLiNER2.5-Decide answers every decision of the training lake on this machine,
recorded `use: train`. A hosted model (Jev, GLiDE, OpenAI) is added as one more answerer recorded `use: eval_only`
until its terms are checked in writing. Every record keeps the question and the text asked, the raw output, the value,
the confidence, the model's revision, the time and the licence.

Sources: [Fastino training](https://docs.fastino.ai/training), [Fastino datasets](https://docs.fastino.ai/concepts/datasets),
[GLiNER2.5-Decide model card](https://huggingface.co/fastino/GLiNER2.5-Decide),
[GLiNER2.5-Decide release](https://datanorth.ai/news/fastino-releases-gliner2-5-decide),
[GLiDE release](https://benchlm.ai/models/fastino-glide), [decision models compared](https://benchlm.ai/decision-models),
[TypeSafe introduction](https://docs.typesafe.ai/introduction), [TypeSafe confidence](https://docs.typesafe.ai/confidence),
[TypeSafe System One](https://docs.typesafe.ai/concepts/system-one),
[Jev pricing](https://aiweekly.co/alerts/typesafe-ais-jev-outputs-numbers-not-text-at-0042m-input),
[Jev data handling](https://innfactory.ai/en/ai-models/typesafe-jev/),
[OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions),
[OpenAI terms of use](https://openai.com/en-GB/policies/terms-of-use/).
