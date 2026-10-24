#!/bin/zsh
# One homework a process. Before each: no heavy job beside the model (store run, archive, a gate: together they swapped
# the disk full on 2026-10-24), and the owner's disk rule (below 2 GB free: unload the model and wait).
# usage: loop.sh LOG python -m dclab_rnd.learning pilot|probe ... --limit 1
LOG=$1; shift
cd "/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D"
while true; do
  while pgrep -f "dclab_rnd.store run|dclab_rnd.archive|run_parallel|check-all|rd-gates" >/dev/null; do
    echo "waiting: a store run, archive or gate is running $(date +%H:%M)" >> $LOG; ollama stop qwen3.5:4b-16k >/dev/null 2>&1; sleep 120
  done
  while [ $(df -g . | tail -1 | awk '{print $4}') -lt 2 ]; do
    echo "disk below 2 GB: model unloaded, waiting $(date +%H:%M)" >> $LOG
    ollama stop qwen3.5:4b-16k >/dev/null 2>&1; sleep 120
  done
  "$@" >> $LOG 2>&1; rc=$?
  echo "process exit $rc at $(date +%H:%M), swap $(sysctl -n vm.swapusage | awk '{print $6}')" >> $LOG
  [ $rc -eq 3 ] || break
done
echo "loop done (exit $rc)" >> $LOG
