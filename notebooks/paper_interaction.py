"""Extract a source-faithful IKEA interaction figure from results archives."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import io
import numpy as np
from PIL import Image
from assembly_world_agent.episode_io import read_episode

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
PAPER = ROOT.parent / 'AssemblyWorldBench'
SELECTED = [7, 32, 53, 56, 76]  # One-based environment call numbers.
FEATURED = [[16, 20], [49, 52], [54, 55], [62]]
STAGES = ['Initial Observation', 'Partial Assembly', 'Inspect End Frames', 'Reorient End Frames', 'Final Observation']


def extract(results=RESULTS, paper=PAPER):
    results, paper = Path(results), Path(paper)
    block = results / 'ikea-manual/gpt-6-astra/ikea-manualbook'
    sample = block / 'samples/Bench--applaro'
    archive = sample / 'final.episode.zip'
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    metric = next(json.loads(l) for l in (block / 'evaluation/metrics.jsonl').read_text().splitlines() if json.loads(l)['sample_id'] == 'Bench/applaro')
    assert sha == metric['episode_sha256']
    episode = read_episode(archive)
    calls = episode['calls']
    assert len(calls) == 76 and all(c['index'] == i for i,c in enumerate(calls))
    initial_state = calls[1]['result']
    assert calls[1]['name'] == 'get_state' and initial_state['index'] == 0
    assert len(initial_state['objects']) == 4
    assert all(o['position'] == [0, 0, 0] and o['quaternion'] == [1, 0, 0, 0]
               for o in initial_state['objects'])
    start = datetime.fromisoformat(calls[0]['timestamp'])
    output = paper / 'fig/interaction'
    (output / 'assets').mkdir(parents=True, exist_ok=True)
    report = dict(sample_id='Bench/applaro', system='GPT-6 Astra + Codex', parts=4, total_calls=len(calls), capture_calls=sum(c['name']=='capture_scene' for c in calls), source_archive=str(archive.relative_to(results)), archive_sha256=sha, extraction_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), selection_reason='Author preferred the first IKEA bench; selected observations cover the initial scene, partial assembly, same-camera end-frame reorientation, and final observation. Not selected as a representative performance estimate.', time_origin=start.isoformat(), time_definition='Wall-clock difference between recorded environment-call timestamps, relative to the first environment call; not model thinking time or total task runtime.', index_definition='#k denotes one-based environment call number k; archive index is k-1. Not an observation count or state index.', scoring={k:metric[k] for k in ('PA','SR','SCD','state_index')}, panels=[], transitions=[], calls=[])
    for number, stage in zip(SELECTED, STAGES):
        call = calls[number-1]
        assert call['name']=='capture_scene' and call['status']=='completed'
        name = next(c['observation'] for c in call['result']['content'] if c['type']=='image')
        raw = episode['files']['observations/'+name]
        image_sha = hashlib.sha256(raw).hexdigest()
        assert image_sha == name.removesuffix('.png')
        filename=f'assets/capture-{number:02d}.png'
        (output/filename).write_bytes(raw)
        # Author-approved display-only background replacement; preserve every other pixel.
        pixels = np.array(Image.open(io.BytesIO(raw)).convert('RGBA'))
        background = np.all(pixels == [24, 24, 37, 255], axis=-1)
        display = pixels.copy()
        display[background] = [220, 224, 230, 255]
        assert np.array_equal(display[~background], pixels[~background])
        display_name = f'assets/display-{number:02d}.png'
        Image.fromarray(display).save(output / display_name)
        elapsed=(datetime.fromisoformat(call['timestamp'])-start).total_seconds()
        tenths=round(elapsed*10)
        label=f'{tenths//600:02d}:{tenths%600/10:04.1f}'
        report['panels'].append(dict(call_number=number, archive_index=number-1, stage=stage, elapsed_seconds=elapsed, wall_time=label, timestamp=call['timestamp'], state_index=call['state_index'], camera=call['result']['camera'], image=filename, image_sha256=image_sha, original_member='observations/'+name, display_image=display_name, display_sha256=hashlib.sha256((output/display_name).read_bytes()).hexdigest(), background_pixel_count=int(background.sum())))
    assert report['panels'][2]['camera']==report['panels'][3]['camera']
    for a,b,selected in zip(SELECTED[:-1],SELECTED[1:],FEATURED):
        assert all(a<n<b for n in selected)
        report['transitions'].append(dict(from_call=a,to_call=b,featured_calls=selected,omitted_call_count=b-a-1-len(selected),all_between=list(range(a+1,b))))
    manual_source = sample / 'manualbook/page-007.png'
    manual_bytes = manual_source.read_bytes()
    manual_name = 'assets/manual-page-007.png'
    (output / manual_name).write_bytes(manual_bytes)
    report['manual'] = {'image':manual_name, 'source':str(manual_source.relative_to(results)),
                        'sha256':hashlib.sha256(manual_bytes).hexdigest(), 'page':7,
                        'steps':[6], 'crop_xywh':[80,70,1090,1430],
                        'selection':'Reference input illustration, not a timestamped manual-reading event.'}
    report['calls']=[{k:c[k] for k in ('index','name','arguments','timestamp','before_index','state_index','status')} for c in calls]
    # These exact pixel rectangles identify the same left end-frame region in both observations.
    report['initial_pose_verification'] = {'call_number':2, 'state_index':0, 'parts':4, 'position':[0,0,0], 'quaternion_wxyz':[1,0,0,0], 'preparation':'Initial scattered poses are baked into mesh vertices in preparation.py.'}
    report['display_processing'] = {'method':'Exact RGBA background replacement only; all other pixels unchanged.', 'source_rgba':[24,24,37,255], 'display_rgba':[220,224,230,255]}
    report['detail_crop']={'panels':[53,56], 'rectangle_xywh':[115,225,225,135], 'note':'Identical source-pixel crop, native PPTX clipping only; full images remain visible.'}
    report['dataset_label'] = 'IKEA-Manual'
    report['purpose_labels'] = ['Orient and Place','Adjust and Inspect','Reorient End Frames','Turn Upright']
    report['additional_rows'] = [
        extract_row(results, output, 'fantastic-breaks', 'fantastic-breaks-none', '00/00017',
                    'Fantastic Breaks', 'evaluation/chamfer-v2/metrics.jsonl',
                    [2,20,25,30,49], [[14],[21,23],[28,29],[44,48]],
                    ['Place Fragment','Adjust Alignment','Close the Gap','Inspect and Adjust']),
        extract_row(results, output, 'assemblybench', 'assemblybench-manualbook', '1047',
                    'AssemblyBench', 'evaluation/metrics.jsonl',
                    [2,34,42,57,61], [[20,28],[37,40],[47,55],[58,59]],
                    ['Orient the Body','Place Components','Inspect the Interior','Adjust and Restore'])]
    (output/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def extract_row(results, output, domain, block_name, sample_id, label, metrics_path, selected, featured, purposes):
    block = results / domain / 'gpt-6-astra' / block_name
    sample = block / 'samples' / sample_id.replace('/', '--')
    archive = sample / 'final.episode.zip'
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    metric = next(json.loads(l) for l in (block / metrics_path).read_text().splitlines()
                  if json.loads(l)['sample_id'] == sample_id)
    assert digest == metric['episode_sha256']
    episode = read_episode(archive)
    calls = episode['calls']
    assert all(c['index'] == i for i, c in enumerate(calls))
    start = datetime.fromisoformat(calls[0]['timestamp'])
    row = dict(dataset_label=label, sample_id=sample_id, system='GPT-6 Astra + Codex',
               source_archive=str(archive.relative_to(results)), archive_sha256=digest,
               total_calls=len(calls), purpose_labels=purposes, panels=[], transitions=[],
               scoring={k:metric[k] for k in ('PA','SR','SCD','state_index')},
               selection_reason='Illustrative archive-selected trajectory with visible intermediate manipulation; not a representative performance estimate.',
               time_origin=start.isoformat(), reference_condition='No visual reference' if domain=='fantastic-breaks' else 'Assembly manual',
               reference_asset_status='Not applicable' if domain=='fantastic-breaks' else 'Manual metadata exists in results/input.json, but page images are not archived in this results sample.')
    for n in selected:
        call = calls[n-1]
        assert call['name']=='capture_scene' and call['status']=='completed'
        member = 'observations/' + next(e['observation'] for e in call['result']['content'] if e['type']=='image')
        raw = episode['files'][member]
        image_digest = hashlib.sha256(raw).hexdigest()
        assert member.endswith(image_digest+'.png')
        filename = f'assets/{domain}-{n:02d}.png'
        (output/filename).write_bytes(raw)
        pixels = np.array(Image.open(io.BytesIO(raw)).convert('RGBA'))
        bg = pixels[0,0].copy()
        assert np.array_equal(bg, [24,24,37,255])
        mask = np.all(pixels==bg, axis=-1)
        display = pixels.copy(); display[mask] = [220,224,230,255]
        assert np.array_equal(display[~mask], pixels[~mask])
        display_name = f'assets/{domain}-display-{n:02d}.png'
        Image.fromarray(display).save(output/display_name)
        elapsed = (datetime.fromisoformat(call['timestamp'])-start).total_seconds()
        tenths = round(elapsed*10)
        row['panels'].append(dict(call_number=n, archive_index=n-1, state_index=call['state_index'],
                                 image=filename, image_sha256=image_digest, original_member=member,
                                 display_image=display_name, display_sha256=hashlib.sha256((output/display_name).read_bytes()).hexdigest(),
                                 background_pixel_count=int(mask.sum()), wall_time=f'{tenths//600:02d}:{tenths%600/10:04.1f}',
                                 elapsed_seconds=elapsed, timestamp=call['timestamp'], camera=call['result']['camera']))
    for a,b,ns in zip(selected[:-1],selected[1:],featured):
        assert all(a<n<b for n in ns)
        assert all(calls[n-1]['status']=='completed' for n in ns)
        row['transitions'].append(dict(from_call=a,to_call=b,featured_calls=ns,
                                      omitted_call_count=b-a-1-len(ns),all_between=list(range(a+1,b))))
    if domain == 'assemblybench':
        row['manual'] = resolve_assemblybench_reference(sample, output)
        row['reference_asset_status'] = 'Last reference step recovered from pinned dataset cache; byte hash verified against results input metadata.'
    row['calls']=[{k:c[k] for k in ('index','name','arguments','timestamp','before_index','state_index','status')} for c in calls]
    return row


def resolve_assemblybench_reference(sample, output):
    """Resolve the author-requested source reference; experiment evidence stays in results."""
    from assembly_world_agent.adapters.assemblybench import reference_pages
    import pyarrow as pa
    import os
    inputs = json.loads((sample/'input.json').read_text())
    sid, revision = inputs['sample_id'], inputs['revision']
    expected = inputs['manual']['pages'][-1]
    cache = ROOT/'notebooks/.cache/paper-analysis/reference-assets'
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache/f'assemblybench-{sid}-final.png'
    if cached.exists():
        raw = cached.read_bytes()
    else:
        dataset_cache = Path(os.environ.get('HF_DATASETS_CACHE', Path.home()/'.cache/huggingface/datasets'))
        pinned = dataset_cache/'AssemblyWorld___assemblybench/default/0.0.0'/revision
        found = None
        for shard in sorted(pinned.glob('assemblybench-full-*.arrow')):
            with pa.memory_map(str(shard), 'r') as source:
                for batch in pa.ipc.open_stream(source):
                    ids = batch.column(batch.schema.get_field_index('object_id')).to_pylist()
                    if sid in ids:
                        idx = ids.index(sid)
                        found = {name:batch.column(batch.schema.get_field_index(name))[idx].as_py()
                                 for name in ('object_id','steps','manual_pages')}
                        break
            if found is not None:
                break
        if found is None:
            raise FileNotFoundError(f'Pinned AssemblyBench reference unavailable for {sid} at revision {revision}')
        page = reference_pages(found, 'manualbook')[-1]
        raw = page['image']['bytes']
        assert hashlib.sha256(raw).hexdigest() == expected['image_sha256']
        cached.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == expected['image_sha256']
    name = f'assets/assemblybench-{sid}-manual-final.png'
    (output/name).write_bytes(raw)
    return {'image':name,'sha256':digest,'dataset':'AssemblyWorld/assemblybench',
            'revision':revision,'sample_id':sid,'page':expected['page'],'step_id':expected['step_id'],
            'source_file':expected['source_file'], 'results_metadata':str((sample/'input.json').relative_to(ROOT/'results')),
            'retrieval':'Read-only pinned dataset Arrow cache; adapter-selected final reference step; exact hash match to results/input.json'}


if __name__=='__main__':
    report=extract()
    print(json.dumps({k:report[k] for k in ('sample_id','total_calls','capture_calls','index_definition','time_definition')},indent=2))
    for p in report['panels']: print(p['call_number'],p['wall_time'],p['stage'])
