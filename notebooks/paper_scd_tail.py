"""Audit the upper tail of the five reported PartNet SCD distributions."""
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
PAPER = ROOT.parent / 'AssemblyWorldBench'
CACHE = ROOT / 'notebooks/.cache/paper-analysis'
BLOCKS = ('chair-none', 'table-none', 'storage-none', 'table-final-image', 'storage-final-image')
LABELS = ('Chair / NR', 'Table / NR', 'Storage / NR', 'Table / IR', 'Storage / IR')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(results=RESULTS, paper=PAPER, cache=CACHE, audit_cases=True):
    """Recompute aggregates; do not reuse unverified caches or rescore geometry."""
    results, paper, cache = map(Path, (results, paper, cache))
    assert results.name == 'results'
    assert not cache.resolve().is_relative_to(results.resolve())
    report = {'definition': 'Sort all scored objects by descending SCD, then sample ID; k=ceil(0.05*N). Contribution is sum(top k SCD)/sum(all SCD). No cases are removed from reported means.', 'rows': [], 'cases': [], 'implementation_sha256': digest(Path(__file__)), 'protocol': 'Saved assembly-evaluation-v2 SCD; no geometry rescoring, clipping, or per-assembly renormalization.'}
    for block, label in zip(BLOCKS, LABELS):
        base = results / 'partnet/gpt-6-astra' / block
        path = base / 'evaluation/metrics.jsonl'
        summary_path = path.with_name('metrics_summary.json')
        summary = json.loads(summary_path.read_text())
        excluded = set(summary.get('excluded_ids', []))
        all_rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        rows = [r for r in all_rows if r.get('status') == 'scored' and r['sample_id'] not in excluded]
        assert rows and all(math.isfinite(r['SCD']) and r['SCD'] >= 0 for r in rows)
        rows.sort(key=lambda r: (-r['SCD'], r['sample_id']))
        n, total = len(rows), sum(r['SCD'] for r in rows)
        mean = total/n
        assert math.isclose(mean, summary['SCD'], rel_tol=1e-9, abs_tol=1e-9)
        k = math.ceil(.05*n)
        report['rows'].append(dict(block=block, label=label, n=n, k=k, fraction=k/n, mean=mean, median=statistics.median(r['SCD'] for r in rows), tail_share=sum(r['SCD'] for r in rows[:k])/total, remainder_mean=sum(r['SCD'] for r in rows[k:])/(n-k), tail_zero_PA=sum(r['PA']==0 for r in rows[:k]), tail_SR_failures=sum(r['SR']==0 for r in rows[:k]), tail=[{key:r[key] for key in ('sample_id','SCD','PA','SR')} for r in rows[:k]], input=str(path.relative_to(results)), metrics_sha256=digest(path), summary_sha256=digest(summary_path)))
        if audit_cases and block == 'storage-final-image':
            import numpy as np
            from assembly_world_agent.episode_io import read_episode
            for row in rows[:k]:
                directory=base/'samples'/row['sample_id']
                episode=directory/'final.episode.zip'
                sha=digest(episode)
                assert sha == row['episode_sha256']
                ep=read_episode(episode)
                count=len(ep['manifest']['objects'])
                def poses(state):
                    return np.array(ep['states'][state]['integration'][1:1+7*count]).reshape(count,7)
                initial=poses(0); final=poses(max(ep['states']))
                displacement=np.linalg.norm(final[:,:3]-initial[:,:3],axis=1)
                rotation_change=np.minimum(np.linalg.norm(final[:,3:]-initial[:,3:],axis=1),np.linalg.norm(final[:,3:]+initial[:,3:],axis=1))
                result=json.loads((directory/'result.json').read_text())
                report['cases'].append(dict(sample_id=row['sample_id'],episode_sha256=sha,parts=count,unchanged_body_positions=int((displacement<=1e-8).sum()),unchanged_poses=int(((displacement<=1e-8)&(rotation_change<=1e-8)).sum()),execution_status=result.get('status')))
    cache.mkdir(parents=True,exist_ok=True)
    (cache/'scd_tail.json').write_text(json.dumps(report,indent=2)+'\n')
    (paper/'provenance/scd_tail.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=[r'\begin{table}[tbp]',r'\centering\footnotesize\setlength{\belowcaptionskip}{4pt}',r'\caption{Concentration of PartNet SCD for Astra. The largest $k=\lceil0.05N\rceil$ values define the upper tail; its contribution is the fraction of summed SCD. Mean and median use all objects; the last column excludes the upper tail only as a diagnostic. SCD uses the same scale as Table~\ref{tab:partnet}.}',r'\label{tab:scd_tail}',r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Setting & $N$ & Mean & Median & $k$ & Contribution (\%) & Remaining mean \\',r'\midrule']
    for r in report['rows']:
        lines.append(f"{r['label']} & {r['n']} & {r['mean']:.2f} & {r['median']:.2f} & {r['k']} & {100*r['tail_share']:.2f} & {r['remainder_mean']:.2f}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (paper/'tab/scd_tail.tex').write_text('\n'.join(lines)+'\n')
    return report


if __name__ == '__main__':
    report=generate()
    for r in report['rows']:
        print(r['label'], r['n'], f"top share={100*r['tail_share']:.2f}%", f"remaining mean={r['remainder_mean']:.2f}")
    print(json.dumps(report['cases'],indent=2))
