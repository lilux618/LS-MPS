#!/usr/bin/env python3
"""Compare candidate rain-window outputs with a CPU reference export."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from collections import defaultdict


def read_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def interp(ref, times, field):
    xs=[float(r['time']) for r in ref]; ys=[float(r[field]) for r in ref]
    out=[]
    for t in times:
        if t<=xs[0]: out.append(ys[0]); continue
        if t>=xs[-1]: out.append(ys[-1]); continue
        k=1
        while xs[k]<t: k+=1
        a=(t-xs[k-1])/(xs[k]-xs[k-1]); out.append(ys[k-1]*(1-a)+ys[k]*a)
    return out


def rel_l2(a,b):
    num=sum((x-y)**2 for x,y in zip(a,b)); den=sum(y*y for y in b)
    return math.sqrt(num/max(den,1e-30))


def rmse(a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/max(len(a),1))


def grouped(rows,key):
    d=defaultdict(list)
    for r in rows: d[r[key]].append(r)
    for v in d.values(): v.sort(key=lambda r: float(r['time']))
    return d


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reference',required=True); ap.add_argument('--candidate',required=True); ap.add_argument('--config',default='config/rain_plate_validation.json'); ap.add_argument('--out',default='outputs/validation_compare')
    a=ap.parse_args(); ref=Path(a.reference); cand=Path(a.candidate); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(a.config).read_text()); tol=cfg['comparison_tolerances']; metrics=[]
    rw=grouped(read_csv(ref/'sampling_windows.csv'),'window'); cw=grouped(read_csv(cand/'sampling_windows.csv'),'window')
    for name in sorted(set(rw)&set(cw)):
        times=[float(r['time']) for r in cw[name]]
        for field,kind,limit in [('coverage_ratio','rmse',tol['coverage_ratio_rmse']),('fluid_volume_m3','rel_l2',tol['fluid_volume_relative_l2']),('film_mean_m','rel_l2',tol['film_mean_relative_l2'])]:
            cv=[float(r[field]) for r in cw[name]]; rv=interp(rw[name],times,field); val=rmse(cv,rv) if kind=='rmse' else rel_l2(cv,rv)
            metrics.append({'scope':name,'metric':field+'_'+kind,'value':val,'tolerance':limit,'pass':val<=limit})
    rs=grouped(read_csv(ref/'flow_sections.csv'),'section'); cs=grouped(read_csv(cand/'flow_sections.csv'),'section')
    for name in sorted(set(rs)&set(cs)):
        rfinal=float(rs[name][-1]['cumulative_volume_m3']); cfinal=float(cs[name][-1]['cumulative_volume_m3']); val=abs(cfinal-rfinal)/max(abs(rfinal),1e-30)
        metrics.append({'scope':name,'metric':'cumulative_flow_relative_error','value':val,'tolerance':tol['cumulative_flow_relative_error'],'pass':val<=tol['cumulative_flow_relative_error']})
    mass=read_csv(cand/'mass_conservation.csv'); val=max(float(r['relative_mass_error']) for r in mass)
    metrics.append({'scope':'global','metric':'max_relative_mass_error','value':val,'tolerance':tol['max_relative_mass_error'],'pass':val<=tol['max_relative_mass_error']})
    with (out/'comparison_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=metrics[0].keys()); w.writeheader(); w.writerows(metrics)
    summary={'reference':str(ref),'candidate':str(cand),'passed':all(r['pass'] for r in metrics),'metrics':metrics}
    (out/'comparison_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2)); raise SystemExit(0 if summary['passed'] else 2)
if __name__=='__main__': main()
