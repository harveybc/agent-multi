"""Common-time mark-to-market scores; no refit, account reset or fee refund."""
import csv
from datetime import datetime
import hashlib
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from tools.paired_checkpoint_diagnostics import digest,read,write

PROTOCOL='window_common_suffix.v1'
CONTEXT_BARS=35


def trace_score(path,initial_cash,timezone,skip):
    if (type(skip) is not int or skip<0 or not math.isfinite(initial_cash)
            or initial_cash<=0):raise ValueError('COMMON_TRACE_ARGUMENTS')
    previous=None;anchor=initial_cash;peak=initial_cash;drawdown=0;equity=None
    count=0;first=None;last=None;days=set();support=hashlib.sha256();zone=ZoneInfo(timezone)
    with Path(path).open(newline='') as stream:
        for index,row in enumerate(csv.DictReader(stream)):
            stamp=datetime.fromisoformat(row['timestamp'].replace('Z','+00:00'))
            value=float(row['equity'])
            if (stamp.tzinfo is None or not math.isfinite(value) or value<=0
                    or previous is not None and stamp<=previous):raise ValueError('COMMON_TRACE_INVALID')
            previous=stamp
            if index<skip:
                anchor=peak=value
                continue
            equity=value;peak=max(peak,value);drawdown=max(drawdown,100*(peak-value)/peak)
            last=stamp.isoformat();first=first or last;count+=1
            days.add(stamp.astimezone(zone).date());support.update((last+'\n').encode())
    if not count:raise ValueError('COMMON_TRACE_EMPTY')
    return dict(growth=math.log(equity/anchor)/len(days),return_fraction=equity/anchor-1,
        max_drawdown_pct=drawdown,anchor_equity=anchor,rows=count,days=len(days),
        first=first,last=last,support_sha256=support.hexdigest(),trace_sha256=digest(path))


def require_equal_support(a,b):
    if any(a[k]!=b[k] for k in ('rows','days','first','last','support_sha256')):
        raise ValueError('COMMON_SUPPORT_MISMATCH')


def score(node,native):
    root=Path(node['root']);contract=read(root/'CONTRACT.json');roles={}
    for task in read(root/'STATUS.json')['tasks']:
        role=task['split']
        if role not in ('train','validation') or role in roles:raise ValueError('COMMON_ROLES')
        for key in ('config','summary','return_trace_file'):
            if digest(task[key])!=task['artifacts'][key]:raise ValueError('COMMON_SOURCE_CHANGED')
        cfg=read(task['config']);summary=read(task['summary'])
        warmup=1+(cfg['window_size']-1)*cfg['observation_stride_bars']
        if warmup>CONTEXT_BARS or cfg['mode']!='inference' and role!='train':
            raise ValueError('COMMON_CONTEXT_TOO_SHORT')
        measured=trace_score(task['return_trace_file'],cfg['initial_cash'],cfg['entry_day_timezone'],CONTEXT_BARS-warmup)
        if measured['rows']!=summary['curated_input_identity']['row_count']-CONTEXT_BARS:
            raise ValueError('COMMON_ROW_COUNT')
        measured['input_manifest_sha256']=contract['splits'][role]['curated_input_manifest_sha256']
        roles[role]=measured
    if set(roles)!={'train','validation'}:raise ValueError('COMMON_ROLES')
    return dict(native,train_growth=roles['train']['growth'],validation_growth=roles['validation']['growth'],
        max_drawdown_pct=roles['validation']['max_drawdown_pct'],common_support=dict(protocol=PROTOCOL,
            context_bars=CONTEXT_BARS,roles=roles,limitation='PRE_SUFFIX_POSITIONS_AND_ACCOUNT_HISTORIES_NOT_RESET'),
        native_growth=dict(train=native['train_growth'],validation=native['validation_growth']))


def publish(spec,state,root):
    from tools import rl_warehouse
    nodes=[dict(name='baseline',**spec['baseline'])]
    nodes.extend(dict(name=t['name'],**t['attempts'][-1]) for t in state['trials'] if t['state']=='CLOSED')
    destination=Path(root)/'common-support-publication'
    publication=read(spec['base_request'])['publication'];rl_warehouse.prepare(publication,destination)
    reports=[]
    from tools.temporal_capacity_sweep import scored_outcome
    for node in nodes:
        result=scored_outcome(spec,node)
        for task in read(Path(node['root'])/'STATUS.json')['tasks']:
            role=task['split'];report=read(task['warehouse_report']);values=result['common_support']['roles'][role]
            identity=':'.join((PROTOCOL,node['contract_sha256'],role,digest(Path(root)/'SPEC.json')))
            key='window-suffix:'+hashlib.sha256(identity.encode()).hexdigest()
            report['tags'].update(source_evaluation_code_commit=report['code_commit'],
                source_evaluation_config_sha256=report['config_sha256'],
                validation_role='DEVELOPMENT_HYPERPARAMETER_SELECTION_NOT_INDEPENDENT_HOLDOUT')
            report.update(experiment_key=key,phase='SAC_FINE_WINDOW_COMMON_SUPPORT',
                experiment_set_key=publication['experiment_set_key']+'-fine-window-common',
                code_commit=spec['runtime'],config_sha256=digest(Path(root)/'SPEC.json'))
            report['tags'].update(common_support_protocol=PROTOCOL,context_bars=str(CONTEXT_BARS),
                support_sha256=values['support_sha256'],source_trace_sha256=values['trace_sha256'],
                source_contract_sha256=node['contract_sha256'],window_size=str(node['architecture']['window_size']),
                scope='DEVELOPMENT_COMMON_TIME_NOT_RESET_ACCOUNT_COUNTERFACTUAL')
            report['metrics']=[dict(metric='common_suffix.'+name,value=values[name],split=role,horizon=None,unit=unit)
                for name,unit in [('growth','log_fraction_per_observed_NY_date'),('return_fraction','fraction'),
                    ('max_drawdown_pct','percent'),('rows','bars'),('days','observed_NY_dates')]]
            path=destination/(node['name']+'-'+role+'.json')
            write(path,dict(report=report,source_unit=key));queued=rl_warehouse.invoke(publication,destination,'enqueue',path)
            reports.append(dict(report=report,report_sha256=queued['report_sha256']))
    request=destination/'pump.json';write(request,{k:publication[k] for k in ('warehouse_url','token_env')})
    receipt=rl_warehouse.invoke(publication,destination,'pump',request)
    write(Path(root)/'COMMON_REPORTS.json',dict(protocol=PROTOCOL,reports=reports))
    receipt.update(spec_sha256=digest(Path(root)/'SPEC.json'),reports_sha256=digest(Path(root)/'COMMON_REPORTS.json'),
        contracts=sorted(node['contract_sha256'] for node in nodes))
    write(Path(root)/'COMMON_PUBLICATION_STATUS.json',receipt)
    return 2 if receipt['warehouse']['pending'] or receipt['warehouse']['mismatch'] else 0


def publication_ready(root,spec,state):
    path=Path(root)/'COMMON_PUBLICATION_STATUS.json'
    if not path.exists():return False
    receipt=read(path)
    expected=sorted([spec['baseline']['contract_sha256']]+[
        t['attempts'][-1]['contract_sha256'] for t in state['trials'] if t['state']=='CLOSED'])
    return (receipt.get('spec_sha256')==digest(Path(root)/'SPEC.json')
        and (Path(root)/'COMMON_REPORTS.json').exists()
        and receipt.get('reports_sha256')==digest(Path(root)/'COMMON_REPORTS.json')
        and receipt.get('contracts')==expected and receipt['warehouse']['pending']==0
        and receipt['warehouse']['mismatch']==0)
