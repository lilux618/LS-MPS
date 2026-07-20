#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); ap.add_argument('--min-valid-ratio',type=float,default=0.25)
    a=ap.parse_args(); out=Path(a.output)
    rows=list(csv.DictReader((out/'metrics.csv').open()))
    summary=json.loads((out/'summary.json').read_text())
    checks=[]
    def add(name, ok, value, limit): checks.append({'check':name,'passed':bool(ok),'value':value,'limit':limit})
    add('mass_balance', summary['max_relative_mass_error'] <= 1e-12, summary['max_relative_mass_error'], '<=1e-12')
    add('invalid_removal', summary['max_invalid_removed_ratio'] <= 0.02, summary['max_invalid_removed_ratio'], '<=0.02')
    add('ppe_all_steps_converged', summary['ppe_converged_steps']==summary['ppe_total_steps'], f"{summary['ppe_converged_steps']}/{summary['ppe_total_steps']}", 'all')
    add('ppe_residual', summary['max_ppe_relative_residual'] <= 1e-5, summary['max_ppe_relative_residual'], '<=1e-5')
    warm=rows[max(0,len(rows)//3):]
    valid=sum(float(r['wls_valid_ratio']) for r in warm)/max(len(warm),1)
    add('warm_wls_valid_ratio', valid >= a.min_valid_ratio, valid, f'>={a.min_valid_ratio}')
    increased=sum(float(r['div_l2_after']) > float(r['div_l2_before'])+1e-10 for r in rows)
    add('projection_non_increasing', increased==0, increased, '0 steps')
    passed=all(x['passed'] for x in checks)
    report={'passed':passed,'checks':checks,'note':'CPU LS-MPS beta numerical-health gate; not experimental validation.'}
    (out/'lsmps_validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    md=['# LS-MPS CPU Beta Validation','',f"Overall: **{'PASS' if passed else 'FAIL'}**",'','| Check | Result | Value | Limit |','|---|---|---:|---|']
    for x in checks: md.append(f"| {x['check']} | {'PASS' if x['passed'] else 'FAIL'} | {x['value']} | {x['limit']} |")
    (out/'lsmps_validation_report.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2)); raise SystemExit(0 if passed else 1)
if __name__=='__main__': main()
