"""Fail-closed full-state resume and exhausted-search contracts for346."""


def validate_resume(cp,identity,source_sha,origin_steps):
    if cp['identity']!=identity or cp['source_sha256']!=source_sha:raise ValueError('RETENTION_LINEAGE')
    if cp['offset']!=16 or cp['searches_used']!=4:raise ValueError('RETENTION_RESUME_BUDGET')
    if [r['offset'] for r in cp['trace']]!=list(range(1,17)):raise ValueError('RETENTION_PREFIX')
    counts=[r['searches_used'] for r in cp['trace']]
    if counts!=sorted(counts) or counts[-1]!=4 or counts[0]<0:raise ValueError('RETENTION_SEARCH_HISTORY')
    if cp['accepted']!=sum(r['accepted'] for r in cp['trace']):raise ValueError('RETENTION_ACCEPTED_COUNT')
    steps=[int(v['step']) for v in cp['optimizer']['state'].values()]
    if not steps or any(s!=origin_steps+cp['accepted'] for s in steps):raise ValueError('RETENTION_OPTIMIZER')


def validate_retention_suffix(rows,start,end):
    if [r['offset'] for r in rows]!=list(range(start+1,end+1)):raise ValueError('RETENTION_SUFFIX_SEQUENCE')
    if any(r['arm']!='ordinary' or r['searches_used']!=4 or 'target_search' in r for r in rows):raise ValueError('RETENTION_EXTRA_SEARCH')
