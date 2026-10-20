# SPDX-License-Identifier: AGPL-3.0-only
# Local cache integration, exported 2026-10-20; see ../../LICENSE.
"""Choose a protected base to reuse/update; similarity never authorizes KV reuse.

Both engines use the same deterministic policy. Only exact tokens plus compatible
runtime can be reused. Other scores decide which of two managed entries to update.
"""


def common_prefix(a,b):
    count=0
    for x,y in zip(a,b):
        if x!=y:break
        count+=1
    return count


def _shingles(tokens,width=5):
    if len(tokens)<width:return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i+width]) for i in range(len(tokens)-width+1)}


def choose_base(tokens,entries,*,runtime=None,source_identity=None,capacity=2):
    """Return {id, mode:exact|create|replace, ambiguous, scores}.

    Entry fields: id, occupied, tokens, runtime, source_identity, valid, last_used.
    An empty slot is filled before replacement. With two occupied slots, stable
    source identity helps with a format update; otherwise global five-token
    similarity plus common-prefix length handles edits near the beginning too.
    A genuine tie updates the least recently used slot, then its stable ID.
    """
    if capacity!=2 or not isinstance(tokens,list) or not tokens:
        raise ValueError('Two-base selection requires a nonempty token vector')
    allowed={'base-1','base-2'}
    records=[e for e in entries if e.get('occupied',True)]
    if len(records)>2 or len({e['id'] for e in records})!=len(records):
        raise ValueError('Invalid two-base catalog')
    if any(e['id'] not in allowed for e in records):raise ValueError('Invalid base ID')
    valid=[e for e in records if e.get('valid',True)
        and (runtime is None or e.get('runtime')==runtime) and e.get('tokens')==tokens]
    if valid:
        winner=min(valid,key=lambda e:e['id'])
        return {'id':winner['id'],'mode':'exact','ambiguous':False,'scores':[]}
    scores=[]
    wanted=_shingles(tokens)
    for entry in records:
        previous=entry.get('tokens') or []
        lcp=common_prefix(tokens,previous)
        old=_shingles(previous)
        dice=2*len(wanted&old)/max(1,len(wanted)+len(old))
        coverage=lcp/max(1,min(len(tokens),len(previous)))
        hint=bool(source_identity is not None and entry.get('source_identity')==source_identity)
        scores.append({'id':entry['id'],'lcp':lcp,'dice':round(dice,9),
            'score':round(.7*dice+.3*coverage,9),'source_match':hint,
            'last_used':entry.get('last_used',0) or 0})
    empty=sorted(allowed-{e['id'] for e in records})
    if empty:return {'id':empty[0],'mode':'create','ambiguous':False,'scores':scores}
    ranked=sorted(scores,key=lambda s:(-int(s['source_match']),-s['score'],s['last_used'],s['id']))
    a,b=ranked
    ambiguous=a['source_match']==b['source_match'] and abs(a['score']-b['score'])<=1e-9
    return {'id':a['id'],'mode':'replace','ambiguous':ambiguous,'scores':scores}
