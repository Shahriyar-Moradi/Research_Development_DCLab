# Data for Record linkage and duplicate detection

| Dataset | Source URL | License | File | SHA-256 |
|---|---|---|---|---|
| DBLP-ACM (2,616 + 2,294 records, 2,224 true matches) | https://dbs.uni-leipzig.de/files/datasets/DBLP-ACM.zip | research benchmark of the University of Leipzig's database group (cite Köpcke, Thor and Rahm, 2010) | `data/downloads/entity_resolution/DBLP-ACM/DBLP-ACM.zip` (gitignored) | e37e7ed8e06722499e6d9e94583fdec279207b13ce6270a37595b6a5d594bc40 |
| Abt-Buy (1,081 + 1,092 records, 1,097 true matches) | https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip | the same benchmark collection | `data/downloads/entity_resolution/Abt-Buy/Abt-Buy.zip` (gitignored) | b9ff2937d97371b7a00b0a97033213bfbdee24e8f1a0718a41ea8364472a8ff8 |
| Retail returns replica (copied bills) | `dclab_core/replicas/retail_returns.py` | invented data (30.9) | generated, seeded | - |

The two benchmarks were downloaded with the owner's yes (2026-10-19); they are read only as data (an isolated Python, nothing
in them executed) and never leave the Mac.
