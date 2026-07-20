#!/usr/bin/env python3
"""Run the single rain-plate case at multiple particle spacings."""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path


def last_rows(path,key):
    with path.open(newline='') as f: rows=list(csv.DictReader(f))
    out={}
    for r in rows: out[r[key]]=r
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--duration',type=float,default=0.18); ap.add_argument('--out',default='outputs/resolution_convergence'); ap.add_argument('--spacings',default='0.014,0.012,0.010')
    a=ap.parse_args(); root=Path(a.out); root.mkdir(parents=True,exist_ok=True); summary=[]
    for h in [float(x) for x in a.spacings.split(',')]:
        case=root/f'h_{h:.4f}'; dt=0.0015*(h/0.012); steps=max(1,round(a.duration/dt))
        subprocess.run([sys.executable,'python/rain_plate_cpu.py','--output',str(case),'--steps',str(steps),'--l0',str(h),'--dt',str(dt),'--validation-config','config/rain_plate_validation.json'],check=True)
        wins=last_rows(case/'sampling_windows.csv','window'); secs=last_rows(case/'flow_sections.csv','section')
        row={'l0':h,'dt':dt,'steps':steps,'duration':steps*dt}
        for n,r in wins.items():
            row[f'{n}.coverage_ratio']=float(r['coverage_ratio']); row[f'{n}.fluid_volume_m3']=float(r['fluid_volume_m3']); row[f'{n}.film_mean_m']=float(r['film_mean_m'])
        for n,r in secs.items(): row[f'{n}.cumulative_volume_m3']=float(r['cumulative_volume_m3'])
        summary.append(row)
    fields=sorted({k for r in summary for k in r})
    with (root/'resolution_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summary)
    (root/'resolution_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
