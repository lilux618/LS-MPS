#!/usr/bin/env python3
"""Structural and invariant checks for rain benchmark outputs."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path

REQUIRED = {
    'metrics.csv': {'step','time','particles','injected_total','outflow_total','invalid_removed_total','mass_error'},
    'sampling_windows.csv': {'time','window','particle_count','fluid_volume_m3','coverage_ratio','film_mean_m','film_p90_m','film_max_m'},
    'flow_sections.csv': {'time','section','crossing_particle_count','instantaneous_flow_rate_m3s','crossing_volume_m3','cumulative_volume_m3'},
    'mass_conservation.csv': {'time','injected_volume_m3','remaining_volume_m3','outlet_volume_m3','invalid_removed_volume_m3','balance_error_m3','relative_mass_error','invalid_removed_ratio'},
}

def rows(path: Path):
    with path.open(newline='') as f: return list(csv.DictReader(f))

def finite(v):
    try: return math.isfinite(float(v))
    except Exception: return False

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); ap.add_argument('--config',default='config/rain_plate_validation.json'); ap.add_argument('--report',default=None)
    a=ap.parse_args(); root=Path(a.output); cfg=json.loads(Path(a.config).read_text()); errors=[]; checks=[]
    data={}
    for name, fields in REQUIRED.items():
        path=root/name
        if not path.exists(): errors.append(f'missing {name}'); continue
        rr=rows(path); data[name]=rr
        if not rr: errors.append(f'empty {name}'); continue
        missing=fields-set(rr[0])
        if missing: errors.append(f'{name} missing columns: {sorted(missing)}')
        bad=sum(not finite(v) for r in rr for k,v in r.items() if k!='window' and k!='section')
        if bad: errors.append(f'{name} contains {bad} non-finite numeric values')
    if 'metrics.csv' in data:
        rr=data['metrics.csv']; times=[float(r['time']) for r in rr]
        if any(b<=a for a,b in zip(times,times[1:])): errors.append('metrics time is not strictly increasing')
        for r in rr:
            identity=int(r['injected_total'])-int(r['outflow_total'])-int(r['invalid_removed_total'])-int(r['particles'])
            if identity!=int(r['mass_error']): errors.append(f"particle identity mismatch at step {r['step']}"); break
        checks.append({'check':'particle_identity','passed':not any('particle identity' in e for e in errors)})
    if 'sampling_windows.csv' in data:
        for r in data['sampling_windows.csv']:
            c=float(r['coverage_ratio'])
            if not 0<=c<=1: errors.append(f"coverage out of range: {r['window']} t={r['time']}"); break
            if float(r['film_p90_m']) > float(r['film_max_m'])+1e-15: errors.append('film p90 exceeds max'); break
    if 'flow_sections.csv' in data:
        by={}
        for r in data['flow_sections.csv']: by.setdefault(r['section'],[]).append(r)
        for sec,rr in by.items():
            rr.sort(key=lambda r:float(r['time'])); vals=[float(r['cumulative_volume_m3']) for r in rr]
            if any(b+1e-18<a for a,b in zip(vals,vals[1:])): errors.append(f'cumulative flow decreases for {sec}')
    if 'mass_conservation.csv' in data:
        rr=data['mass_conservation.csv']; max_mass=max(float(r['relative_mass_error']) for r in rr); max_invalid=max(float(r['invalid_removed_ratio']) for r in rr)
        mass_tol=float(cfg['comparison_tolerances']['max_relative_mass_error'])
        invalid_tol=float(cfg['comparison_tolerances'].get('max_invalid_removed_ratio',0.02))
        if max_mass>mass_tol: errors.append(f'mass error {max_mass:.6g} > {mass_tol}')
        if max_invalid>invalid_tol: errors.append(f'invalid removal ratio {max_invalid:.6g} > {invalid_tol}')
        checks += [{'check':'mass_conservation','value':max_mass,'tolerance':mass_tol,'passed':max_mass<=mass_tol},{'check':'invalid_removal','value':max_invalid,'tolerance':invalid_tol,'passed':max_invalid<=invalid_tol}]
    summary={'output':str(root),'passed':not errors,'errors':errors,'checks':checks}
    report=Path(a.report) if a.report else root/'validation_report.json'; report.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    md=['# Rain Benchmark Validation Report','',f"**Status:** {'PASS' if summary['passed'] else 'FAIL'}",'',f"Output: `{root}`",'']
    if errors: md += ['## Errors','']+[f'- {e}' for e in errors]
    else: md += ['All structural, finite-value, conservation, monotonicity and range checks passed.']
    report.with_suffix('.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2)); raise SystemExit(0 if summary['passed'] else 2)
if __name__=='__main__': main()
