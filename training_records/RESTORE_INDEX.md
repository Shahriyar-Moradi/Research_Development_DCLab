# DCLab: everything moved to the ADATA drive, and how to bring it back

Start here. Written 2026-11-20. The same file is on the drive as `/Volumes/ADATA PRO/DCLAB_RESTORE_INDEX.md` and on the
Mac as `training_records/RESTORE_INDEX.md`. Plug in the drive (it mounts as `ADATA PRO`), open a terminal in the
repository folder, and run the command of the row you need. Nothing here is ever deleted from the drive.

Every item was copied, checked file by file against the Mac copy, indexed on both sides and read back before the Mac copy
was removed. A restore test on 2026-11-20 (snapshot `20261112T162722Z`) brought back 136 of 136 files with matching
SHA-256.

## 1. Training snapshots (`make archive`), 43 of them

- Detailed list, one row per snapshot with its file and its check: `training_records/snapshots/SNAPSHOT_LOCATIONS.md`
  (Mac) and `/Volumes/ADATA PRO/dclab_training_records/SNAPSHOT_LOCATIONS.md` (drive).
- On the Mac still: the newest, `20261119T180610Z` (code reads it), and `20261119T115941Z` (an incomplete snapshot from
  an archive stopped when the disk was full; its content is in the next snapshots).
- Bring one back (the tar named in its row; a snapshot of a day with several has its time in the name):

      tar -xf "/Volumes/ADATA PRO/dclab_training_records/training_snapshot_<YYYYMMDD or YYYYMMDDTHHMMSSZ>.tar" -C training_records/snapshots

- The seven oldest (2026-10-17 to 2026-10-18) are in a disk image:

      hdiutil attach -readonly "/Volumes/ADATA PRO/DCLab/dclab_snapshots.sparsebundle"
      cp -R "/Volumes/<mounted name>/snapshots/<snapshot>" training_records/snapshots/

- Check a restore: each snapshot's `MANIFEST.json` lists every file with its SHA-256; `<tar>.verify.json` beside each tar
  is the check made when it was moved.

## 2. Finished record folders (`scripts/offload_folders.py`), in `/Volumes/ADATA PRO/dclab_offload/`

| What | Size | Bring back (in the repository folder) |
|---|---|---|
| Old live tests and journey plays (2026-10-23 to 10-29) | 0.8 GB | `tar -xf "/Volumes/ADATA PRO/dclab_offload/agent_runs_old_records_20261102.tar" -C .` |
| Experiments: twin, e2e, fraud decisions, bake-off | 1.2 GB | `tar -xf "/Volumes/ADATA PRO/dclab_offload/agent_runs_experiments_20261103.tar" -C .` |
| The training store as of 2026-11-03 | 5.1 GB | `tar -xf "/Volumes/ADATA PRO/dclab_offload/store_backup_20261103.tar" -C .` |
| Finished campaigns: search v2-v3, time decisions v1-v4 | 0.8 GB | `tar -xf "/Volumes/ADATA PRO/dclab_offload/finished_campaigns_20261103.tar" -C .` |
| Backup of `agent_runs` and `research` (2026-11-03) | 3.9 GB | `tar -xf "/Volumes/ADATA PRO/dclab_offload/agent_runs_research_backup_20261103.tar" -C .` |
| Old database backup (2026-10-16) | 0.1 GB | `tar -xf "/Volumes/ADATA PRO/dclab_offload/old_db_backup_20261016.tar" -C .` |
| Raw development transcripts (to 2026-11-04) | 1.8 GB | `tar -xf "/Volumes/ADATA PRO/dclab_offload/dev_trajectories_raw_20261104.tar" -C .` |

Details and checks: `dclab_offload/OFFLOAD_INDEX.md` on the drive, `agent_runs/OFFLOAD_INDEX.md` on the Mac; each tar's
`<name>.verify.json` beside it.

## 3. Local AI models (Ollama), in `/Volumes/ADATA PRO/dclab_offload/`

| What | Size | Bring back |
|---|---|---|
| `bge-m3:567m`, `qwen3-embedding:0.6b` (lost the memory-embedding bake-off) | 1.8 GB | `tar -xf "/Volumes/ADATA PRO/dclab_offload/ollama_unused_models_20261103.tar" -C /tmp && cp -n /tmp/ollama_models/blobs/* ~/.ollama/models/blobs/ && cp -Rn /tmp/ollama_models/manifests/ ~/.ollama/models/manifests/` (or `ollama pull` each) |
| `qwen3.5:4b` (partial: blobs verified, two manifests not) and its 16k Modelfile | 4.0 GB | easiest: `ollama pull qwen3.5:4b`, then `ollama create qwen3.5:4b-16k -f "/Volumes/ADATA PRO/dclab_offload/ollama_qwen35_4b-16k.Modelfile"` |

Details: `dclab_offload/OLLAMA_OFFLOAD_INDEX.json`.

## 4. Docker Desktop's disk (2026-11-04)

`/Volumes/ADATA PRO/dclab_offload/docker_disk_20261104.raw.gz` (2.8 GB; its SHA-256 was checked). Docker Desktop starts
with an empty disk without it. To bring it back: quit Docker Desktop, then

    gunzip -c "/Volumes/ADATA PRO/dclab_offload/docker_disk_20261104.raw.gz" > ~/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw

## 5. Older copies (before the indexed moves)

- `/Volumes/ADATA PRO/DCLab/`: a copy of `training_records/` (store, datasets, test runs, logs) and the disk image of
  the seven oldest snapshots (section 1). Copy back what you need with `cp -R`.
- `/Volumes/ADATA PRO/dclab_data_22OCT/`: zips of `agent_runs`, `lake` and `research` as of 2026-10-22. Bring back:
  `unzip "/Volumes/ADATA PRO/dclab_data_22OCT/<name>.zip" -d /tmp/restore` and copy what you need.
- `/Volumes/ADATA PRO/DCLab_backup_2026-10-22/`: empty.

## When something new moves

Snapshots: `python scripts/offload_snapshots.py --keep-hours 0 --remove` (copies, verifies, indexes, reads back, then
removes; never the newest). Folders: `python scripts/offload_folders.py NAME FOLDER... --remove`. Then add a line here.
