"""Small deterministic geometry-only pilot; not a population comparison."""
from pathlib import Path
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
BLOCKS = {
    'AssemblyBench': 'assemblybench/gpt-6-astra/assemblybench-manualbook',
    'IKEA-Manual': 'ikea-manual/gpt-6-astra/ikea-manualbook',
    'PartNet Chair': 'partnet/gpt-6-astra/chair-none',
    'PartNet Table': 'partnet/gpt-6-astra/table-none',
    'PartNet Storage': 'partnet/gpt-6-astra/storage-none',
}


def measure(path):
    sizes = []
    with zipfile.ZipFile(path) as archive:
        xml = ET.fromstring(archive.read('world/model.xml'))
        meshes = {m.attrib['name']: m for m in xml.findall('./asset/mesh')}
        for body in xml.findall('./worldbody/body'):
            geoms = body.findall('geom')
            assert len(geoms) == 1 and geoms[0].get('type') == 'mesh'
            mesh = meshes[geoms[0].attrib['mesh']]
            vertices = np.array([list(map(float, line.split()[1:4])) for line in archive.read('world/' + mesh.attrib['file']).decode().splitlines() if line.startswith('v ')])
            vertices = np.unique(vertices, axis=0) * np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            centered = vertices - vertices.mean(axis=0)
            _, axes = np.linalg.eigh(centered.T @ centered)
            size = float(np.linalg.norm(np.ptp(centered @ axes, axis=0)))
            assert np.isfinite(size) and size > 0
            sizes.append(size)
    return sizes


def generate():
    report = {'selection': 'First three available archives per block ranked by SHA256(sample directory name); independent of scores. PartNet is stratified by category. Exploratory, unequal source counts; no population inference.',
              'metric': 'Largest / smallest per-part PCA bounding-box diagonal, using all unique OBJ vertices and XML mesh scale. Uniform whole-object normalization cancels in the ratio. Vertex-based PCA is tessellation-dependent; validate with surface sampling in the full study.',
              'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases': []}
    for label, block in BLOCKS.items():
        paths = sorted((RESULTS / block / 'samples').glob('*/final.episode.zip'), key=lambda p: hashlib.sha256(p.parent.name.encode()).hexdigest())[:3]
        for path in paths:
            sizes = measure(path)
            row = dict(dataset=label, sample_id=path.parent.name, parts=len(sizes), ratio=max(sizes)/min(sizes), part_diagonals=sizes, archive=str(path.relative_to(RESULTS)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            report['cases'].append(row)
            print(label, row['sample_id'], row['parts'], round(row['ratio'], 2))
    cache = ROOT / 'notebooks/.cache/paper-analysis'
    cache.mkdir(parents=True, exist_ok=True)
    (cache / 'part_scale_pilot.json').write_text(json.dumps(report, indent=2) + '\n')
    return report



def generate_full():
    """Audit every scored unique object in the five source blocks, excluding reported exclusions."""
    from concurrent.futures import ThreadPoolExecutor
    report = {'selection': 'All scored objects in the five source-level blocks; exclusions follow metrics_summary.json. No duplication across PartNet reference conditions. Descriptive statistics characterize evaluated subsets, not original source populations.',
              'metric': 'Largest / smallest PCA bounding-box diagonal of unique vertices per supplied part, with XML mesh scale. Invariant to shared uniform scaling. Vertex-PCA is tessellation-dependent.',
              'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'inputs': [], 'cases': [], 'missing': [], 'summaries': []}
    jobs = []
    for label, block in BLOCKS.items():
        base = RESULTS / block
        metrics_path = base / 'evaluation/metrics.jsonl'
        summary_path = base / 'evaluation/metrics_summary.json'
        summary = json.loads(summary_path.read_text())
        excluded = set(map(str, summary.get('excluded_ids', [])))
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        report['inputs'].append(dict(block=block, metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest(), summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(), excluded_ids=sorted(excluded)))
        for row in rows:
            if str(row['sample_id']) in excluded:
                continue
            if row.get('status') != 'scored':
                report['missing'].append(dict(dataset=label, sample_id=row['sample_id'], reason='not scored'))
                continue
            jobs.append((label, block, row))

    def process(job):
        label, block, score = job
        path = RESULTS / block / 'samples' / str(score['sample_id']).replace('/', '--') / 'final.episode.zip'
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if score.get('episode_sha256'):
            assert sha == score['episode_sha256'], str(path)
        sizes = measure(path)
        return dict(dataset=label, sample_id=score['sample_id'], parts=len(sizes), ratio=max(sizes)/min(sizes), part_diagonals=sizes, PA=score['PA'], SR=score['SR'], archive=str(path.relative_to(RESULTS)), sha256=sha)

    with ThreadPoolExecutor(max_workers=3) as pool:
        for i, row in enumerate(pool.map(process, jobs), 1):
            report['cases'].append(row)
            if i % 100 == 0:
                print(f'Analyzed {i}/{len(jobs)} objects', flush=True)
    rng = np.random.default_rng(0)
    for label in [*BLOCKS, 'PartNet pooled']:
        cases = [r for r in report['cases'] if r['dataset'] == label or label == 'PartNet pooled' and r['dataset'].startswith('PartNet ')]
        values = np.array([r['ratio'] for r in cases])
        boots = np.median(rng.choice(values, size=(10000, len(values)), replace=True), axis=1)
        quantiles = np.quantile(values, [.25, .5, .75, .9, .95])
        report['summaries'].append(dict(dataset=label, n=len(values), q25=quantiles[0], median=quantiles[1], q75=quantiles[2], p90=quantiles[3], p95=quantiles[4], maximum=float(values.max()), median_ci95=np.quantile(boots, [.025, .975]).tolist(), fraction_over_10=float(np.mean(values > 10)), fraction_over_20=float(np.mean(values > 20))))
    report['uncertainty'] = 'Percentile object bootstrap for the median, 10000 resamples, NumPy default_rng seed 0; descriptive resampling uncertainty for the evaluated set. Pooled PartNet weights objects, not categories.'
    cache = ROOT / 'notebooks/.cache/paper-analysis'
    cache.mkdir(parents=True, exist_ok=True)
    (cache / 'part_scale_full.json').write_text(json.dumps(report, indent=2) + '\n')
    paper = ROOT.parent / 'AssemblyWorldBench'
    (paper / 'provenance/part_scale_full.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = [r'\begin{table}[tbp]', r'\centering\footnotesize', r'\caption{Within-object part-size ratios on the evaluated source-level sets. Each ratio divides the largest part PCA bounding-box diagonal by the smallest. PartNet weights objects equally across the three categories.}', r'\label{tab:part_scale}', r'\begin{tabular}{lrrrrr}', r'\toprule', r'Dataset & $N$ & Median & Interquartile range & 90th percentile & Ratio $>10$ (\%) \\', r'\midrule']
    for row in report['summaries']:
        if row['dataset'] not in ('AssemblyBench', 'IKEA-Manual', 'PartNet pooled'):
            continue
        display_name = 'PartNet' if row['dataset'] == 'PartNet pooled' else row['dataset']
        lines.append(f"{display_name} & {row['n']} & {row['median']:.2f} & {row['q25']:.2f}--{row['q75']:.2f} & {row['p90']:.2f} & {100*row['fraction_over_10']:.1f}" + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (paper / 'tab/part_scale.tex').write_text('\n'.join(lines) + '\n')
    print(json.dumps(report['summaries'], indent=2), flush=True)
    return report


if __name__ == '__main__':
    generate_full()
