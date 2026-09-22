"""Generate paper evaluation-set coverage from read-only results records."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
OUTPUT = ROOT.parent / 'AssemblyWorldBench' / 'tab' / 'dataset_coverage.tex'
CACHE = ROOT / 'notebooks' / '.cache' / 'paper-analysis' / 'dataset_coverage.json'


def generate():
    hashes = {}

    def read(path):
        data = path.read_bytes()
        hashes[str(path.relative_to(RESULTS))] = hashlib.sha256(data).hexdigest()
        return data.decode()

    def count(relative, exclude=()):
        path = RESULTS / relative
        rows = [json.loads(line) for line in read(path).splitlines() if line.strip()]
        rows = [row for row in rows if row['sample_id'] not in exclude]
        counts = {}
        run = path.parents[1] if path.parent.name == 'evaluation' else path.parents[2]
        for row in rows:
            sid = row['sample_id']
            assert sid not in counts
            n = len(row.get('parts', []))
            if not n:
                record = json.loads(read(run / 'samples' / sid / 'input.json'))
                n = record['parts']
            assert isinstance(n, int) and n > 0
            counts[sid] = n
        return counts

    specs = [
        ('Benchmark subsets', 'PartNet', 'assemblyworldbench/gpt-6-astra/partnet-none/evaluation/metrics.jsonl', 'None / assembled-furniture image', ()),
        ('Benchmark subsets', 'IKEA-Manual', 'assemblyworldbench/gpt-6-astra/ikea-manualbook/evaluation/metrics.jsonl', 'Real IKEA instructions', ()),
        ('Benchmark subsets', 'AssemblyBench', 'assemblyworldbench/gpt-6-astra/assemblybench-manualbook/evaluation/metrics.jsonl', 'Rendered step diagrams', ()),
        ('Benchmark subsets', 'Fantastic Breaks', 'assemblyworldbench/gpt-6-astra/fantastic-breaks-none/evaluation/metrics.jsonl', 'None', ()),
        ('Larger evaluations', 'PartNet chair', 'partnet/gpt-6-astra/chair-none/evaluation/metrics.jsonl', 'None', ()),
        ('Larger evaluations', 'PartNet table', 'partnet/gpt-6-astra/table-none/evaluation/metrics.jsonl', 'None / assembled-furniture image', ()),
        ('Larger evaluations', 'PartNet storage', 'partnet/gpt-6-astra/storage-none/evaluation/metrics.jsonl', 'None / assembled-furniture image / manual', ()),
        ('Larger evaluations', 'PartNet chair (subset)', 'partnet/gpt-6-astra/chair-manualbook/evaluation/metrics.jsonl', 'Manual', ()),
        ('Larger evaluations', 'PartNet table (subset)', 'partnet/gpt-6-astra/table-manualbook/evaluation/metrics.jsonl', 'Manual', ()),
        ('Larger evaluations', 'IKEA-Manual', 'ikea-manual/gpt-6-astra/ikea-manualbook/evaluation/metrics.jsonl', 'Real IKEA instructions', ()),
        ('Larger evaluations', 'AssemblyBench', 'assemblybench/gpt-6-astra/assemblybench-manualbook/evaluation/metrics.jsonl', 'Rendered step diagrams', ('6772',)),
        ('Larger evaluations', 'Fantastic Breaks', 'fantastic-breaks/gpt-6-astra/fantastic-breaks-none/evaluation/chamfer-v2/metrics.jsonl', 'None', ()),
    ]
    records = []
    for panel, name, relative, reference, excluded in specs:
        counts = count(relative, excluded)
        values = list(counts.values())
        records.append(dict(panel=panel, name=name, source=relative, reference=reference,
                            shapes=len(values), parts=sum(values), minimum=min(values), maximum=max(values),
                            shape_parts=counts, excluded=list(excluded)))
    # Verify combined reference rows and the relationship to the larger shape pools.
    for index, alternate in [(0, 'assemblyworldbench/gpt-6-astra/partnet-final-image/evaluation/metrics.jsonl'),
                             (5, 'partnet/gpt-6-astra/table-final-image/evaluation/metrics.jsonl'),
                             (6, 'partnet/gpt-6-astra/storage-final-image/evaluation/metrics.jsonl'),
                             (6, 'partnet/gpt-6-astra/storage-manualbook/evaluation/metrics.jsonl')]:
        assert records[index]['shape_parts'] == count(alternate)
    for small, large in [(1, 9), (2, 10), (3, 11)]:
        assert all(records[large]['shape_parts'].get(k) == v for k, v in records[small]['shape_parts'].items())
    pool = {k: v for i in (4, 5, 6) for k, v in records[i]['shape_parts'].items()}
    assert all(pool.get(k) == v for k, v in records[0]['shape_parts'].items())
    lines = [r'\begin{table}[tbp]', r'\centering\footnotesize\setlength{\belowcaptionskip}{4pt}',
             r'\caption{Dataset coverage in the reported evaluations. Upper rows are the subsets used for all eight systems; lower rows are the larger source-level evaluations. Counts describe these evaluation sets, not entire original datasets. Parts are counted once per shape, not once per reference condition or system. The paired PartNet benchmark conditions give 40 tasks from 20 shapes. Larger PartNet rows overlap; they should not be summed. AssemblyBench excludes one object from its 280-object run.}',
             r'\label{tab:dataset_coverage}',
             r'\begin{NiceTabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrl@{}}', r'\toprule',
             r'Dataset / evaluation set & Shapes & Total parts & Parts/shape & Reference \\', r'\midrule']
    previous = None
    for row in records:
        if row['panel'] != previous:
            if previous is not None:
                lines.append(r'\midrule')
            lines.append(r'\multicolumn{5}{l}{\textit{' + row['panel'] + r'}} \\')
            previous = row['panel']
        span = str(row['minimum']) if row['minimum'] == row['maximum'] else f"{row['minimum']}--{row['maximum']}"
        lines.append(f"{row['name']} & {row['shapes']:,} & {row['parts']:,} & {span} & {row['reference']} " + r'\\')
    lines += [r'\bottomrule', r'\end{NiceTabular*}', r'\end{table}']
    OUTPUT.write_text('\n'.join(lines) + '\n')
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(dict(input_hashes=hashes, records=records), indent=2) + '\n')
    for row in records:
        print(row['name'], row['shapes'], row['parts'], row['minimum'], row['maximum'])
    return records


if __name__ == '__main__':
    generate()
