#!/bin/bash
# Snapshot of a debate benchmark: progress, rate, ETA, accuracy, round mix.
# Usage: scripts/agents/watch_run.sh [label]
LABEL="${1:-dissent_adjudicated_balanced90_v1}"
DIR=reports/debate
LOG="$DIR/$LABEL.run.log"
CK="$DIR/$LABEL.checkpoint.jsonl"
PID=$(pgrep -f "evaluate_debate_pubmedqa[.]py" | head -1)

TOTAL=$(grep -oE '^\[[0-9]+/[0-9]+\]' "$LOG" 2>/dev/null | head -1 | tr -d '[]' | cut -d/ -f2)
TOTAL=${TOTAL:-0}
DONE=$(grep -cE '^\[[0-9]+/[0-9]+\] (PASS|FAIL)' "$LOG" 2>/dev/null)
[ -f "$CK" ] && DONE=$(wc -l < "$CK")

if [ -z "$PID" ]; then
  [ -f "$DIR/$LABEL.json" ] && echo "run: UKONCZONY -> $DIR/$LABEL.json" || echo "run: NIE DZIALA (przerwany)"
else
  echo "run: dziala (pid $PID, $(( $(ps -p $PID -o etimes=|tr -d ' ')/60 )) min)"
fi
echo "postep: $DONE / ${TOTAL:-?}   timeouty: $(grep -c 'did not respond' "$LOG" 2>/dev/null)   bledy supervisora: $(grep -c 'Failed to parse SupervisorModerationOutput' "$LOG" 2>/dev/null)"

python3 - "$LOG" "$CK" "$PID" "$TOTAL" "$DIR/$LABEL.json" <<'PY'
import json, sys, re, datetime, subprocess, os
from collections import Counter
log, ck, pid, total, final = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4] or 0), sys.argv[5]

rows = []
if os.path.exists(ck):
    rows = [json.loads(l) for l in open(ck)]
elif os.path.exists(final):
    rows = json.load(open(final))["cases"]

if pid and total:
    n_done = len(rows) or len(re.findall(r'^\[\d+/\d+\] (?:PASS|FAIL)', open(log).read(), re.M))
    if n_done:
        el = int(subprocess.check_output(["ps","-p",pid,"-o","etimes="]).strip())
        r = el / n_done
        left = (total - n_done) * r
        print(f"tempo: {r:.1f}s/case -> pozostalo {left/60:.0f} min, koniec ok. "
              + (datetime.datetime.now()+datetime.timedelta(seconds=left)).strftime('%H:%M'))

if not rows:
    # bez checkpointu: policz z logu (etykiety niedostepne, tylko PASS/FAIL)
    txt = open(log).read() if os.path.exists(log) else ""
    p = len(re.findall(r'\] PASS ', txt)); f = len(re.findall(r'\] FAIL ', txt))
    if p+f: print(f"\nPASS {p} / FAIL {f}  -> accuracy biezaca {p/(p+f):.3f}")
    rr = Counter(int(m) for m in re.findall(r'rounds=(\d+)', txt))
    if rr: print("rundy:", dict(sorted(rr.items())))
    raise SystemExit

print()
print("klasa    n   debata    BERT      R1")
for lab in ['yes','no','maybe']:
    s=[x for x in rows if x['expected_label']==lab]
    if s: print(f"  {lab:6s}{len(s):4d}   {sum(1 for x in s if x['label_pass'])/len(s):.3f}   "
                f"{sum(1 for x in s if x.get('biolinkbert_pass'))/len(s):.3f}   {sum(1 for x in s if x['round1_pass'])/len(s):.3f}")
n=len(rows)
print(f"  RAZEM {n:4d}   {sum(1 for x in rows if x['label_pass'])/n:.3f}   "
      f"{sum(1 for x in rows if x.get('biolinkbert_pass'))/n:.3f}   {sum(1 for x in rows if x['round1_pass'])/n:.3f}")
ch=[x for x in rows if x['round1_vote_label']!=x['predicted_label']]
if ch:
    print(f"\nzmiany R1->final: {len(ch)} ({len(ch)/n:.0%}) | naprawione "
          f"{sum(1 for x in ch if x['label_pass'] and not x['round1_pass'])} | zepsute "
          f"{sum(1 for x in ch if x['round1_pass'] and not x['label_pass'])}")
print("rundy:", dict(sorted(Counter(x['rounds_run'] for x in rows).items())))
PY
echo
nvidia-smi --query-gpu=index,name,utilization.gpu --format=csv,noheader | awk -F', ' '{printf "  GPU%s %-12s %s\n",$1,$2,$3}'
