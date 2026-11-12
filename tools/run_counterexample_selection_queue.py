"""Finite counterexample contrasts followed by conditional cumulative selection."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys

from app.frozen_recipe_inputs import file_sha
from tools.run_bayesian_head_contrast import verify_done
from tools.run_discrete_curve_probe import read,write,now


def run(a):
    def call(tool,*args):
        write(a.root/'PROGRESS.json',dict(stage=tool,arguments=list(map(str,args)),at=now()))
        subprocess.run([sys.executable,'tools/'+tool,*map(str,args)],check=True)
    chosen=None
    # Fixed priority, not ranking by validation. The fallback relaxes only the
    # extra counterexample margin floor, never the original FIT/value budgets.
    for mode in ('floor','boundary'):
        out=a.root/mode
        call('run_account_counterexamples.py','--parent',a.parent,'--bundle',a.bundle,
             '--floor',a.floor,'--probe',a.probe,'--diagnostic',a.diagnostic,
             '--output',out,'--guard-mode',mode,'--expected-device-name',a.expected_device_name)
        result=read(out/'RESULT.json');verify_done(out,result['identity'])
        if result['qualified_for_selection']:
            chosen=out;break
        if mode=='floor' and not any(r['added_constraints'] for r in result['rounds']):
            break
    if chosen is None:
        write(a.root/'QUEUE_RESULT.json',dict(status='COMPLETE_NO_QUALIFIED_NEW_CONTEXT',
            selection_started=False,source_preserved=True,at=now()))
        return
    template=read(a.selection_template)
    selection=a.root/'selection'
    if not (selection/'SPEC.private.json').exists():
        call('run_cumulative_mask_pass.py','prepare','--root',selection,'--parent',a.parent,
            '--baseline',chosen/'bundle','--archive',template['archive'],
            '--archive-step',template['archive_step'],'--archive-receipt-sha',
            template['source_files'][template['archive']+'/RESULT.json'],
            '--original-selection',template['original_selection'],'--device','cuda')
    call('run_cumulative_mask_pass.py','run','--root',selection)
    call('report_cumulative_mask_pass.py','--root',selection,'--output',a.root/'selection_report')
    call('export_cumulative_sac_actor.py','--selection',selection,'--output',a.root/'export')
    selected=read(selection/'RESULT.json');export=read(a.root/'export/RESULT.json')
    verify_done(a.root/'export',export['identity'])
    write(a.root/'QUEUE_RESULT.json',dict(status='COMPLETE_SELECTION_AND_PHYSICAL_EXPORT',
        selected_worker=chosen.name,selection_identity=selected['identity'],export_identity=export['identity'],
        template_sha256=file_sha(a.selection_template),selection_started=True,model_promoted=False,at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','parent','bundle','floor','probe','diagnostic','selection-template'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    with (a.root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.root/'QUEUE_FAILED.json',dict(error=repr(exc),at=now()));raise
