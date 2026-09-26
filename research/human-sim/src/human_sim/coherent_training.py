"""TRAIN-only fitting for the continuous residual runtime.

The geometry and polynomial basis are shared with runtime; no unpublished
pilot modules or worktree paths are required. Ridge=1 fixes the historical
integrated curve-space fit. Admission reads only TRAIN movement and recipient
identity metadata to enforce independent player/map-family components.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from . import coherent_execution as runtime
from .execution_exploratory import _read_decoded
from .execution_training import transition_route
from .io import load_map_plan

MAX_NODES=runtime.MAX_NODES
MAX_COEFFICIENTS=runtime.MAX_COEFFICIENTS
RIDGE=1.0
MAX_CONTEXTS=400
MAX_PER_PLAYER=2
MAX_PER_REPLAY=2
MAX_PER_COMPONENT=2
MAX_GAP_MS=40.0
geometry=runtime._geometry
features=runtime._features
normalize_coefficients=runtime._normalize

def make_context(row,times,positions,events,centers):
    secants=np.diff(centers,axis=0)/(np.diff(events)[:,None]/1000.)
    context=runtime._context(events,centers,np.array([secants[0],secants[-1]]),np.zeros((2,2)))
    common=np.unique(np.r_[np.arange(events[0],events[-1],2.),events[-1]])
    context.update(times=times, observed=positions, common=common,
        common_matrix=runtime._design(common,context['nodes']),
        identity={k:row[k] for k in ('split','player','map_family','map','replay_sha256','first_transition_index','span')})
    return context

def time_weights(times):
    delta=np.diff(times)
    if len(delta)==0 or np.any(delta<=0): raise ValueError('Invalid physical-time grid')
    weights=np.r_[delta[0]/2,(delta[:-1]+delta[1:])/2,delta[-1]/2]
    return weights/(times[-1]-times[0])

def initial_state(context):
    coefficients=context['baseline']
    return np.array([coefficients[0],coefficients[1]/.1,coefficients[2]/.01])

def prior(context,incoming):
    if np.linalg.norm(incoming[0]-context['centers'][0])>1e-6:
        raise ValueError('Incoming P conflicts with frozen task contact; unsupported')
    result=context['baseline'].copy()
    result[0]=incoming[0]; result[1]=incoming[1]*.1; result[2]=incoming[2]*.01
    return result

def state_bank(context,g,amplitudes):
    initial=initial_state(context); bank=[initial]
    for order in (1,2):
        for dimension in range(2):
            for sign in (-1,1):
                incoming=initial.copy()
                delta=np.zeros(2); delta[dimension]=sign*amplitudes[order-1]*g['length']/(.1**order)
                incoming[order]+=delta@g['rotation']
                bank.append(incoming)
    return bank

def padded_design(context,g):
    design=np.zeros((len(context['common']),MAX_COEFFICIENTS))
    design[:,g['mapping']]=context['common_matrix']
    return design

def roughness(context,g):
    x,weights=np.polynomial.legendre.leggauss(16); times=[]; scales=[]
    nodes=context['nodes']
    for left,right in zip(nodes[:-1],nodes[1:]):
        times.extend(left+(x+1)*(right-left)/2)
        scales.extend(np.sqrt(weights*(right-left)/(2*(nodes[-1]-nodes[0]))))
    actual=runtime._design(np.asarray(times),nodes,4,3)*np.asarray(scales)[:,None]*.1**3
    result=np.zeros((len(times),MAX_COEFFICIENTS)); result[:,g['mapping']]=actual
    return result

def refit_bank(context,g,bank,free_initial=False):
    design=padded_design(context,g); weights=time_weights(context['common'])
    observed=np.column_stack([np.interp(context['common'],context['times'],context['observed'][:,d]) for d in range(2)])
    observed=(observed-g['origin'])@g['rotation'].T/g['length']
    penalty=roughness(context,g)
    mask=g['mask'].copy()
    if free_initial: mask[1:3]=1
    free=np.flatnonzero(mask)
    matrix=np.vstack((np.sqrt(weights)[:,None]*design[:,free],np.sqrt(1e-4)*penalty[:,free]))
    column_scale=np.maximum(np.linalg.norm(matrix,axis=0),1e-12)
    scaled=matrix/column_scale
    priors=[normalize_coefficients(prior(context,state),context,g) for state in bank]
    rhs=np.column_stack([np.vstack((np.sqrt(weights)[:,None]*(observed-design@base),-np.sqrt(1e-4)*penalty@base)) for base in priors])
    solution=np.linalg.lstsq(scaled,rhs,rcond=1e-11)[0]/column_scale[:,None]
    targets=[]; errors=[]
    for i,base in enumerate(priors):
        target=base.copy(); target[free]+=solution[:,2*i:2*i+2]
        targets.append(target)
        errors.append(float(g['length']*np.sqrt(np.sum(weights*np.sum((design@target-observed)**2,axis=1)))))
    return np.asarray(targets),errors,float(np.linalg.cond(scaled))

def train_joint(contexts,amplitudes):
    geometries=[geometry(context) for context in contexts]
    banks=[state_bank(context,g,amplitudes) for context,g in zip(contexts,geometries)]
    raw_features=[np.array([features(context,state,g) for state in bank]) for context,g,bank in zip(contexts,geometries,banks)]
    all_features=np.vstack(raw_features); mean=all_features.mean(axis=0); scale=np.maximum(all_features.std(axis=0),1e-6)
    loss_scale=float(np.median([g['length'] for g in geometries]))
    k=len(mean)+1; dimension=MAX_COEFFICIENTS*k
    gram=np.zeros((dimension,dimension)); right=np.zeros((dimension,2)); fit_rows=[]
    for index,(context,g,bank,raw) in enumerate(zip(contexts,geometries,banks,raw_features)):
        normalized=np.column_stack((np.ones(len(raw)),(raw-mean)/scale))
        targets,errors,condition=refit_bank(context,g,bank)
        _,oracle_errors,_=refit_bank(context,g,[initial_state(context)],free_initial=True)
        design=padded_design(context,g)*g['mask']
        weights=time_weights(context['common'])
        local_gram=design.T@(weights[:,None]*design)*(g['length']/loss_scale)**2
        priors=np.array([normalize_coefficients(prior(context,state),context,g) for state in bank])
        delta=targets-priors
        feature_gram=normalized.T@normalized/len(bank)
        gram+=np.kron(local_gram,feature_gram)
        for axis in range(2):
            target_feature=delta[:,:,axis].T@normalized/len(bank)
            right[:,axis]+=(local_gram@target_feature).ravel()
        fit_rows.append(dict(identity=context['identity'],oracle_free_initial_rmse_px=oracle_errors[0],
            supplied_initial_target_rmse_px=errors,target_fit_degradation_px=[e-oracle_errors[0] for e in errors],target_condition=condition))
        if (index+1)%20==0: print(f'TARGET FIT {index+1}/{len(contexts)}',flush=True)
    # Explicit positive ridge gives a convex, regularized curve-space problem.
    diagonal=np.maximum(np.diag(gram),1e-10)
    column_scale=np.sqrt(diagonal+RIDGE)
    system=(gram+RIDGE*np.eye(dimension))/column_scale[:,None]/column_scale[None,:]
    factor=cho_factor(system,lower=True,check_finite=True)
    beta=cho_solve(factor,right/column_scale[:,None])/column_scale[:,None]
    weights=beta.reshape(MAX_COEFFICIENTS,k,2)
    model=dict(feature_mean=mean,feature_scale=scale,weights=weights,loss_scale_px=loss_scale,ridge=RIDGE,
        max_nodes=MAX_NODES,amplitudes=amplitudes,solver='column-scaled Cholesky of positive-ridge integrated curve-space normal system',
        regularized_condition_upper_bound=float(np.linalg.norm(gram,ord=1)/RIDGE+1))
    return model,fit_rows

def components(records):
    parent=list(range(len(records)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,row in enumerate(records):
        for key in ('player','replay_sha256','map_family'):
            token=(key,row[key])
            if token in seen:
                a,b=root(i),root(seen[token])
                if a!=b:parent[b]=a
            else:seen[token]=i
    groups=defaultdict(list)
    for i,row in enumerate(records):groups[root(i)].append((i,row))
    result={}
    for values in groups.values():
        tokens=sorted(f"{k}:{r[k]}" for _,r in values for k in ('player','replay_sha256','map_family'))
        cid=hashlib.sha256('|'.join(tokens).encode()).hexdigest()[:16]
        result[cid]=values
    return result

def admission(manifest,source,recipient_rows):
    records=[dict(r,manifest_index=i) for i,r in enumerate(manifest['records']) if r['split']=='train']
    groups=components(records);recipient_tokens={(k,r[k]) for r in recipient_rows for k in ('player','replay_sha256','map_family')}
    blocked={cid for cid,items in groups.items() if any((k,r[k]) in recipient_tokens for _,r in items for k in ('player','replay_sha256','map_family'))}
    counts=Counter();examples=defaultdict(list);accepted=[];player_count=Counter();replay_count=Counter();component_count=Counter();used_intervals=defaultdict(list)
    decoded_root=source/'decoded';map_root=source/'map-plans'
    ordered=[]
    for cid,items in groups.items():
        for index,row in items:ordered.append((index,cid,row))
    ordered.sort(key=lambda x:(x[0],x[2]['replay_sha256'],x[2]['map']))
    cache={}
    for manifest_index,cid,row in ordered:
        if cid in blocked:counts['blocked_transitive_recipient_component']+=1;continue
        replay=row['replay_sha256'];map_hash=row['map'];dp=decoded_root/f'{replay}.ndjson';mp=map_root/f'{map_hash}.map.ndjson.gz'
        if not dp.exists() or not mp.exists():counts['missing_cached_source_record']+=1;continue
        try:
            if replay not in cache:
                header,frames=_read_decoded(dp);cache[replay]=(header,np.asarray([f['time_ms'] for f in frames],float),np.asarray([[f['x'],f['y']] for f in frames],float))
            header,all_times,all_pos=cache[replay];plan=load_map_plan(mp)
        except (ValueError,OSError) as error:counts['unreadable_cached_source_record']+=1;continue
        mods=header.get('mods',[]) if isinstance(header,dict) else []
        if not isinstance(header,dict) or header.get('beatmap_md5')!=map_hash:counts['decoded_map_identity_mismatch']+=1;continue
        if mods not in ([],['CL']):counts['speed_or_time_affecting_mod_record']+=1;continue
        first=1
        while first<len(plan.objects)-2:
            counts['geometry_window_attempts']+=1
            try:routes=[transition_route(plan,first+i,skill_group='intermediate',record_id=f'curve:{replay}:{first}') for i in range(3)]
            except ValueError as error:
                reason=str(error);counts[reason]+=1
                if len(examples[reason])<3:examples[reason].append(dict(replay_sha256=replay,map=map_hash,first_transition_index=first))
                first+=1;continue
            events=np.asarray([routes[0].start_time_ms,*[r.start_time_ms+r.duration_ms for r in routes]],float);start,end=events[0],events[-1]
            objects=plan.objects[first-1:first+3];centers=np.asarray([[o.position.x,o.position.y] for o in objects],float)
            inside=(all_times>=start)&(all_times<=end);native_times=all_times[inside]
            reason=None
            if len(native_times)<5 or all_times[0]>start or all_times[-1]<end:reason='insufficient_source_coverage'
            elif len(native_times)>1 and np.max(np.diff(native_times))>MAX_GAP_MS:reason='source_gap_over_40ms'
            elif not np.all(np.diff(events)>0):reason='nonpositive_event_time'
            elif len(runtime._expand_nodes(events)[0])>MAX_NODES:reason='expanded_node_capacity_over_7'
            if reason:
                counts[reason]+=1
                if len(examples[reason])<3:examples[reason].append(dict(replay_sha256=replay,map=map_hash,first_transition_index=first))
                first+=4;continue
            counts['quality_eligible_nonoverlapping_windows']+=1
            interval=(float(start),float(end))
            if any(not (interval[1]<a or interval[0]>b) for a,b in used_intervals[replay]):reason='overlap_with_selected_replay_window'
            elif player_count[row['player']]>=MAX_PER_PLAYER:reason='player_cap'
            elif replay_count[replay]>=MAX_PER_REPLAY:reason='replay_cap'
            elif component_count[cid]>=MAX_PER_COMPONENT:reason='component_cap'
            elif len(accepted)>=MAX_CONTEXTS:reason='global_context_cap'
            if reason:
                counts[reason]+=1
                if len(examples[reason])<3:examples[reason].append(dict(replay_sha256=replay,map=map_hash,first_transition_index=first))
                first+=4;continue
            sample_times=np.unique(np.r_[native_times,events]);sample_pos=np.column_stack([np.interp(sample_times,all_times,all_pos[:,d]) for d in range(2)])
            identity=dict(split='expanded_train',player=row['player'],map_family=row['map_family'],map=map_hash,replay_sha256=replay,first_transition_index=first,span=3,manifest_index=manifest_index,component_id=cid,mods=mods,source_gap_max_ms=float(np.max(np.diff(native_times))) if len(native_times)>1 else 0.,source_samples=len(native_times),event_start_ms=float(start),event_end_ms=float(end))
            accepted.append(dict(identity=identity,times=sample_times,positions=sample_pos,events=events,centers=centers,decoded_path=str(dp.resolve()),map_path=str(mp.resolve())))
            player_count[row['player']]+=1;replay_count[replay]+=1;component_count[cid]+=1;used_intervals[replay].append(interval);counts['admitted']+=1
            first+=4
    return accepted,dict(records=len(records),components=len(groups),blocked_components=len(blocked),eligible_components=len(groups)-len(blocked),counts=dict(counts),reason_examples=dict(examples),admitted_contexts=len(accepted),admitted_components=len({x['identity']['component_id'] for x in accepted}),players=len({x['identity']['player'] for x in accepted}),replays=len({x['identity']['replay_sha256'] for x in accepted}),families=len({x['identity']['map_family'] for x in accepted}),caps=dict(contexts=MAX_CONTEXTS,per_player=MAX_PER_PLAYER,per_replay=MAX_PER_REPLAY,per_component=MAX_PER_COMPONENT))

def bank_amplitudes(contexts):
    magnitudes={1:[],2:[]}
    for context in contexts:
        g=geometry(context)
        times=np.concatenate([left+(right-left)*np.array([.25,.5,.75]) for left,right in zip(context['nodes'][:-1],context['nodes'][1:])])
        for order in (1,2):
            values=runtime._design(times,context['nodes'],4,order)@context['baseline']
            magnitudes[order].extend(np.linalg.norm(values,axis=1)*.1**order/g['length'])
    return np.array([.25*np.quantile(magnitudes[order],.75) for order in (1,2)])
