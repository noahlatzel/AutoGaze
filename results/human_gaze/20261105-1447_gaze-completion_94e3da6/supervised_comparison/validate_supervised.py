from pathlib import Path
import json, sys, numpy as np
sys.path.insert(0,'/home/stud/latn/master-thesis/AutoGaze')
from scripts.human_gaze.curate_supervised_k16_comparison import load_frozen_analysis_config, validate_analysis_config, expected_action_paths
from autogaze.human_gaze.checkpoint_provenance import load_checkpoint_inventory, verify_supervised_checkpoint, verify_rl_checkpoint
from autogaze.datasets.av_gaze_stavis import read_manifest
from autogaze.human_gaze.supervised_analysis import load_action_export, cell_mass_for_records, aggregate_frame_values, selected_mass, source_stratified_video_bootstrap_delta, seed_summary, sha256_file
repo=Path('/home/stud/latn/master-thesis/AutoGaze');root=Path(__file__).resolve().parent.parent
cfg=load_frozen_analysis_config(repo/'experiments/human_gaze/configs/supervised_k16_restart_analysis.yaml');validate_analysis_config(cfg)
inv=load_checkpoint_inventory(repo/cfg['checkpoint_provenance']['inventory'],expected_sha256=cfg['checkpoint_provenance']['inventory_sha256'],repository_root=repo)
metrics=json.loads((root/'supervised_curation/metrics.json').read_text());gate=json.loads((root/'supervised_curation/gate.json').read_text())
assert metrics['quality_only_amendment'] and not gate['hlvid_admitted']
assert metrics['resources']['status']=='withheld_in_quality_only_amendment'
seeds=cfg['posthoc_seed_policy']['included_base_seeds'];assert metrics['analysis_base_seeds']==seeds
records=[r for r in read_manifest(cfg['data']['manifest']) if r['split']=='val'];mass=cell_mass_for_records(records,np.load(cfg['data']['cell_mass'],mmap_mode='r'))
off=json.loads((repo/cfg['offcenter_subgroup']['manifest']).read_text());offids={r['clip_id'] for r in off['clips']};assert len(offids)==110
reports={};checks=[]
for (method,seed,step),directory in sorted(expected_action_paths(cfg,root/'supervised_actions').items()):
 if method=='supervised':_,p=verify_supervised_checkpoint(inv,training_root=Path(cfg['paths']['supervised_training_root']),base_seed=seed,cumulative_update=step)
 else:_,p=verify_rl_checkpoint(inv,base_seed=seed,cumulative_update=step)
 m,a=load_action_export(directory,records,expected_manifest_sha256=cfg['data']['manifest_sha256'],expected_cell_mass_sha256=cfg['data']['cell_mass_sha256'],expected_method=method,expected_base_seed=seed,expected_training_seed=seed+100000,expected_cumulative_update=step,expected_checkpoint_provenance=p)
 vals=selected_mass(mass,a)
 full=aggregate_frame_values(records,vals);sub=aggregate_frame_values(records,vals,eligible_clip_ids=offids)
 row=next(r for r in metrics['per_seed_checkpoint_rows'] if r['method']==method and r['base_seed']==seed and r['cumulative_update']==step)
 assert row['coverage_full_macro']==full['macro_source_mean']
 assert row['coverage_offcenter_macro']==sub['macro_source_mean']
 reports[(method,seed,step)]={'full_validation':full,'offcenter_top_quartile':sub}
 checks.append({'method':method,'seed':seed,'step':step,'action_sha256':sha256_file(directory/'actions.jsonl'),'provenance_verified':True,'clips':len(a)})
 print(method,seed,step,'PASS',flush=True)
for subset in ['full_validation','offcenter_top_quartile']:
 left=[reports[('supervised',s,20000)][subset] for s in seeds];right=[reports[('rl',s,20000)][subset] for s in seeds]
 b=source_stratified_video_bootstrap_delta(left,right,expected_seed_count=5,iterations=10000,seed=20260910)
 assert b==metrics['paired_video_bootstrap'][subset]
for method,key in [('supervised','supervised_fixed_checkpoints'),('rl','rl_existing_actual_logged_updates')]:
 for row in metrics['convergence'][key]:
  assert row['base_clip_presentations']==row['cumulative_update']*4
  assert row['nominal_training_trajectory_action_rows']==row['base_clip_presentations']*16*16*(1 if method=='supervised' else 4)
assert metrics['convergence']['supervised_fixed_checkpoints'][-1]['cumulative_update']==20000
assert metrics['convergence']['rl_existing_actual_logged_updates'][-1]['cumulative_update']==20000
result={'status':'pass','checks':checks,'paired_bootstrap_intervals_exact_reproduction':True,'frozen_offcenter_clips':110,'matching_base_clip_exposure':80000,'excluded_seed':440830,'hlvid_admitted':False,'training_resource_totals':'withheld','metadata_correction_for_promotion':'Inherited convergence.warning final sentence says scheduler totals include all attempts. Correct to complete scheduler totals unavailable; no numerical changes.'}
(root/'postrun-validation/supervised-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(metrics['endpoint_coverage'],indent=2));print(json.dumps(metrics['paired_video_bootstrap'],indent=2));print('ALL VERIFIED')
