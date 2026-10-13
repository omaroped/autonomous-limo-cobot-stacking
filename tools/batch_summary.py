#!/usr/bin/env python3
import glob
import math
import os
import re
import statistics as st
import sys

PATTERNS = {
    'perception_x': (r'\[perception check\].*?error x ([+-]?[\d.]+) mm', 'mm'),
    'perception_y': (r'\[perception check\].*?error x [+-][\d.]+ mm, y ([+-]?[\d.]+) mm', 'mm'),
    'perception_z': (r'\[perception check\].*?z ([+-]?[\d.]+) mm', 'mm'),
    'jaw_centering': (r'between-jaws=([+-]?[\d.]+) mm', 'mm'),
    'along_faces': (r'along-faces=([+-]?[\d.]+) mm', 'mm'),
    'unpin_drift': (r'\[unpin drift\] base moved ([\d.]+) mm', 'mm'),
    'place_err_xy': (r'err_xy=([\d.]+) m', 'm->mm'),
    'cycle_time': (r'cycle \d+ done:.*?t=([\d.]+)s', 's'),
}

def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def stage_of(txt):
    if not re.search(r'Nav2 goal accepted', txt):
        if re.search(r'Nav2 goal rejected after|NavigateToPose action server not available', txt):
            return 'nav2 did not start'
        return 'no navigation (other)'
    last = re.findall(r'cycle \d+ done: pick=(\w+) retain=(\w+) place=(\w+)', txt)
    if not last:
        return 'no result line (timeout/crash)'
    pick, retain, place = last[-1]
    if place == 'True':
        return 'placed'
    if pick != 'True':
        return 'pick leg'
    if retain != 'True':
        return 'lost in transport'
    if not re.search(r'Nav2 arrived — docking to place table', txt):
        return 'nav2 transit (place leg)'
    if not re.search(r'Place dock complete via AprilTag', txt):
        return 'place dock'
    return 'placement (missed or aborted)'

def stages(logs, startup_retries):
    rows = []
    for f in logs:
        txt = open(f, errors='ignore').read()
        last = re.findall(r'cycle \d+ done: pick=(\w+) retain=(\w+) place=(\w+)', txt)
        pb = re.findall(r'Phase B: centred ✓ (?:lateral offset|residual) ([+-]?\d+) mm', txt)
        pbo = re.findall(r'off the normal line by ([+-]?\d+) mm', txt)
        dock = re.findall(r'Phase C: docked at range=([\d.]+)', txt)
        stall = re.findall(r'Phase C: no progress for [\d.]+ s at range=([\d.]+)', txt)
        err = re.findall(r'err_xy=([\d.]+) m', txt)
        rows.append(dict(
            run=os.path.basename(f), stage=stage_of(txt),
            started=bool(re.search(r'Nav2 goal accepted', txt)),
            pick=bool(last) and last[-1][0] == 'True',
            retain=bool(last) and last[-1][1] == 'True',
            arrived=bool(re.search(r'Nav2 arrived — docking to place table', txt)),
            tagdock=bool(re.search(r'Place dock complete via AprilTag', txt)),
            placed=bool(last) and last[-1][2] == 'True',
            note=' '.join(filter(None, [
                f'B first {pbo[0]} mm' if pbo else '',
                f'B centred {pb[-1]} mm' if pb else '',
                f'C docked {dock[-1]}' if dock else '',
                f'C stalled {stall[-1]}' if stall else '',
                f'err_xy {float(err[-1]) * 1000:.0f} mm' if err else '']))))
    print(f'\n{"run":<12}{"ended at":<32}notes')
    print('-' * 90)
    for r in rows:
        print(f'{r["run"]:<12}{r["stage"]:<32}{r["note"]}')
    n = len(rows)
    chain = [('Nav2 started', 'started', None), ('grasp | started', 'pick', 'started'),
             ('retained | grasped', 'retain', 'pick'), ('arrived | retained', 'arrived', 'retain'),
             ('tag dock | arrived', 'tagdock', 'arrived'), ('placed | tag dock', 'placed', 'tagdock'),
             ('placed, end to end', 'placed', None)]
    print(f'\n{"stage (conditional)":<24}{"k/n":>8}{"rate":>8}{"95% CI":>14}')
    print('-' * 54)
    for name, key, given in chain:
        pool = [r for r in rows if (given is None or r[given])]
        if given == 'tagdock':
            k = sum(1 for r in pool if r['placed'])
        else:
            k = sum(1 for r in pool if r[key])
        m = len(pool)
        lo, hi = wilson(k, m)
        print(f'{name:<24}{f"{k}/{m}":>8}{(k / m * 100 if m else 0):>7.0f}%{f"{lo*100:.0f}-{hi*100:.0f}%":>14}')
    if startup_retries:
        print(f'\nplus {startup_retries} start-up attempt(s) retried by batch_runs.sh '
              f'(Nav2 never active; logs run_*.startupfail*.log)')

def main(d):
    logs = sorted(f for f in glob.glob(os.path.join(d, 'run_*.log'))
                  if re.search(r'run_\d+\.log$', f))
    startup_retries = len(glob.glob(os.path.join(d, 'run_*.startupfail*.log')))
    if not logs:
        print(f'no run_*.log in {d}')
        return 1

    vals = {k: [] for k in PATTERNS}
    picks = places = retains = tag_docks = 0

    for f in logs:
        txt = open(f, errors='ignore').read()
        for key, (pat, unit) in PATTERNS.items():
            m = re.findall(pat, txt, re.S)
            if m:
                v = float(m[-1])
                vals[key].append(v * 1000 if unit == 'm->mm' else v)
        last = re.findall(r'cycle \d+ done: pick=(\w+) retain=(\w+) place=(\w+)', txt)
        if last:
            p, r, pl = last[-1]
            picks += p == 'True'
            retains += r == 'True'
            places += pl == 'True'
        if re.search(r'Place dock complete via AprilTag', txt):
            tag_docks += 1

    n = len(logs)
    print(f'\n{n} runs in {d}\n')
    print(f'{"stage":<16}{"n":>4}{"mean":>10}{"sd":>9}{"min":>9}{"max":>9}')
    print('-' * 57)
    for k in PATTERNS:
        v = vals[k]
        if not v:
            continue
        sd = st.stdev(v) if len(v) > 1 else 0.0
        print(f'{k:<16}{len(v):>4}{st.mean(v):>10.2f}{sd:>9.2f}{min(v):>9.2f}{max(v):>9.2f}')

    print(f'\n{"outcome":<16}{"k/n":>10}{"rate":>9}{"95% CI":>18}')
    print('-' * 55)
    for name, k in (('pick', picks), ('retain', retains), ('place', places),
                    ('dock via tag', tag_docks)):
        lo, hi = wilson(k, n)
        print(f'{name:<16}{f"{k}/{n}":>10}{k / n * 100:>8.0f}%{f"{lo*100:.0f}-{hi*100:.0f}%":>18}')
    print('\nCI assumes independent, comparable trials. Report it; at n=20 the bare '
          'rate overstates\nwhat the data supports (18/20 = 90 % is really 70-97 %).')
    stages(logs, startup_retries)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else '.'))
