"""Small LaTeX tables generated from the notebook analysis outputs."""

import itertools

import pandas as pd
import paper_analysis as pa


def table(name, caption, label, headers, rows, alignment=None):
    alignment = alignment or "l" + "r" * (len(headers) - 1)
    body = "\n".join(" & ".join(map(str, r)) + r" \\" for r in rows)
    text = (
        r"\begin{table}[tbp]"
        + "\n"
        + r"\centering\small\setlength{\belowcaptionskip}{4pt}"
        + "\n"
        + r"\caption{"
        + caption
        + "}\n"
        + r"\label{"
        + label
        + "}\n"
        + r"\begin{NiceTabular*}{\linewidth}{@{\extracolsep{\fill}}"
        + alignment
        + r"@{}}"
        + "\n"
        + r"\toprule"
        + "\n"
        + " & ".join(headers)
        + r" \\"
        + "\n"
        + r"\midrule"
        + "\n"
        + body
        + "\n"
        + r"\bottomrule"
        + "\n"
        + r"\end{NiceTabular*}"
        + "\n"
        + r"\end{table}"
        + "\n"
    )
    (pa.PAPER / "tab" / f"{name}.tex").write_text(text)


def ci(bounds, scale=100):
    return f"[{scale * bounds[0]:.1f}, {scale * bounds[1]:.1f}]"


def probability(value):
    return r"$<0.001$" if value < 0.001 else f"{value:.3f}"


def build(frame):
    result = pa.read_json(pa.CACHE / "results_summary.json")
    rows = result["systems"]
    table(
        "runtime_coverage",
        "Coverage among 100 configured evaluations per system. Timeouts follow the explicit execution status; other failures are counted separately. An execution failure can still have a scoreable partial export. Cost records are counts, not success rates.",
        "tab:runtime_coverage",
        ["System", "Archives", "Scores", "Timeouts", "Other failed", "Cost records"],
        [
            [
                pa.SYSTEMS[r["system"]],
                r["archives"],
                r["scored"],
                r["timeouts"],
                r["other_failed"],
                r["cost_n"],
            ]
            for r in rows
        ],
    )
    table(
        "analysis_overview",
        "Source-weighted benchmark quality and pointwise paired-shape bootstrap intervals (percent). Every system has 100 scored exports; CLI failures may retain scoreable partial scenes.",
        "tab:analysis_overview",
        ["System", "SR", "95\\% CI", "PA", "95\\% CI", "CLI failures"],
        [
            [
                pa.SYSTEMS[r["system"]],
                f"{100 * r['SR']:.1f}",
                ci(r["SR_ci"]),
                f"{100 * r['PA']:.1f}",
                ci(r["PA_ci"]),
                r["runtime_failed"],
            ]
            for r in rows
        ],
    )
    paired = []
    for a, b in itertools.combinations(pa.SYSTEMS, 2):
        sr = next(
            x for x in result["paired"] if x["a"] == a and x["b"] == b and x["metric"] == "SR"
        )
        pr = next(
            x for x in result["paired"] if x["a"] == a and x["b"] == b and x["metric"] == "PA"
        )
        paired.append(
            [
                pa.SHORT_SYSTEMS[a],
                pa.SHORT_SYSTEMS[b],
                f"{100 * sr['delta']:+.1f} {ci(sr['ci'])}",
                probability(sr["p_holm"]),
                f"{100 * pr['delta']:+.1f} {ci(pr['ci'])}",
                probability(pr["p_holm"]),
            ]
        )
    table(
        "analysis_paired",
        "All system differences (A minus B, percentage points), with pointwise paired-shape 95\\% intervals. Randomization-test $p$ values use Holm correction across all 56 system-pair/metric comparisons; intervals remain pointwise.",
        "tab:analysis_paired",
        [
            "A",
            "B",
            r"$\Delta$SR [95\% CI]",
            r"$p_{\rm Holm}$",
            r"$\Delta$PA [95\% CI]",
            r"$p_{\rm Holm}$",
        ],
        paired,
        "llrrrr",
    )
    refs = []
    for s in pa.SYSTEMS:
        d = next(x for x in result["reference"] if x["system"] == s and x["metric"] == "PA")
        sr = next(x for x in result["reference"] if x["system"] == s and x["metric"] == "SR")
        refs.append(
            [
                pa.SYSTEMS[s],
                f"{100 * d['delta']:+.1f}",
                ci(d["ci"]),
                f"{d['positive']}/{d['negative']}",
                f"{sr['positive']}/{sr['negative']}",
            ]
        )
    table(
        "analysis_reference",
        "Paired final-image minus no-reference results on the same 20 PartNet shapes. Up/down counts exclude ties; SR counts are discordant successes. PA intervals are pointwise percentile bootstrap intervals.",
        "tab:analysis_reference",
        ["System", r"$\Delta$PA (pp)", "95\\% CI", "PA up/down", "SR up/down"],
        refs,
    )
    counts = []
    for b, label in zip(pa.BLOCKS, pa.BLOCK_LABELS):
        f = frame[(frame.system == "gpt-6-astra") & (frame.block == b)]
        counts.append(
            [label] + [str(int((f.band == band).sum())) for band in ["low", "mid", "high", "all"]]
        )
    table(
        "complexity_counts",
        "Frozen part-count band sizes per system. PartNet shapes appear in both reference conditions; band definitions are source-specific.",
        "tab:complexity_counts",
        ["Block", "Low", "Middle", "High", "All"],
        counts,
    )
    calibration = pa.read_json(pa.CACHE / "calibration.json")
    table(
        "calibration",
        "Self-reported completion versus offline all-parts success. Missing final reports are excluded from report counts, not assumed to be unable. Precision is successes among completed claims.",
        "tab:calibration",
        ["System", "Reports", "Completed", "True successes", "Precision"],
        [
            [
                pa.SYSTEMS[r["system"]],
                r["reports"],
                r["completed"],
                r["successes"],
                f"{r['precision']:.2f}" if r["precision"] is not None else "--",
            ]
            for r in calibration
        ],
    )
    if (pa.CACHE / "failure_summary.json").exists():
        failures = pa.read_json(pa.CACHE / "failure_summary.json")
        equal = pd.read_csv(pa.CACHE / "failure_shape_equal.csv").set_index("system")
        table(
            "failure_counts",
            "Geometric diagnostic denominators and translation-dominated error fractions. Part pooling weights incorrect parts equally; shape pooling first averages within each failed shape. Categories follow the ordered classifier.",
            "tab:failure_counts",
            ["System", "Failed shapes", "Wrong parts", "Part pooled (\\%)", "Shape pooled (\\%)"],
            [
                [
                    pa.SYSTEMS[r["system"]],
                    r["failed_shapes"],
                    r["wrong_parts"],
                    f"{100 * r['Translation dominated']:.1f}",
                    f"{100 * equal.loc[r['system'], 'Translation dominated']:.1f}",
                ]
                for r in failures
            ],
        )
    # Never insert the report's unsupported standard-GARF or hybrid numbers.
    if (pa.CACHE / "part_diagnostics.csv").exists():
        parts = pd.read_csv(pa.CACHE / "part_diagnostics.csv")
        errors = parts[parts.category != "Correct"]
        by_source = []
        for s in pa.SYSTEMS:
            cells = [pa.SYSTEMS[s]]
            for block in pa.BLOCKS:
                g = errors[(errors.system == s) & (errors.block == block)]
                cells.append(
                    f"{100 * (g.category == 'Translation dominated').mean():.1f} ({len(g)})"
                    if len(g)
                    else "-- (0)"
                )
            by_source.append(cells)
        table(
            "failure_by_source",
            "Translation-dominated fraction by benchmark block. Cells show percent of incorrect parts, followed by the number of incorrect parts in parentheses; -- denotes an empty denominator. These are conditional part-pooled diagnostics, not source-weighted Overall performance.",
            "tab:failure_by_source",
            ["System", "PN / NR", "PN / image", "IKEA", "AB", "FB"],
            by_source,
        )
    sensitivity_path = pa.CACHE / "failure_sensitivity.json"
    if sensitivity_path.exists():
        sensitivity = pd.DataFrame(pa.read_json(sensitivity_path))
        sensitivity_rows = []
        for s in pa.SYSTEMS:
            g = sensitivity[sensitivity.system == s]
            base = g[(g.centroid_threshold == 0.1) & (g.centered_cd_threshold == 0.01)].iloc[0]
            sensitivity_rows.append(
                [
                    pa.SYSTEMS[s],
                    f"{100 * base.translation_fraction:.1f}",
                    f"{100 * g.translation_fraction.min():.1f}",
                    f"{100 * g.translation_fraction.max():.1f}",
                ]
            )
        table(
            "failure_sensitivity",
            "Translation-dominated fraction among incorrect parts (percent): default and range over nine classification settings. Centroid cutoffs are 0.05, 0.1, and 0.2; centered-CD cutoffs are 0.005, 0.01, and 0.02. Correctness, never-moved precedence, and the far-displacement cutoff stay fixed. Ranges are sensitivity summaries, not confidence intervals.",
            "tab:failure_sensitivity",
            ["System", "Default", "Minimum", "Maximum"],
            sensitivity_rows,
        )
    table(
        "hybrid",
        "Matched refinement comparison (pending). No standard-GARF or hybrid per-object outputs are available in the verified result package. The agent-only diagnostic in Table~\\ref{tab:fantastic} is not silently treated as the matched control.",
        "tab:hybrid",
        ["Initialization / system", "PA", "CD", "Pose error", "Cost"],
        [
            ["Agent only (matched)", r"\pending", r"\pending", r"\pending", r"\pending"],
            ["Standard GARF", r"\pending", r"\pending", r"\pending", r"\pending"],
            ["Agent + GARF", r"\pending", r"\pending", r"\pending", r"\pending"],
        ],
    )
    audit_path = pa.CACHE / "regression_audit.json"
    if audit_path.exists():
        audit = pd.DataFrame(pa.read_json(audit_path))
        audit_rows = []
        for system in pa.SYSTEMS:
            g = audit[audit.system == system] if not audit.empty else audit
            audit_rows.append(
                [
                    pa.SYSTEMS[system],
                    len(g),
                    int(g.matching_changes.gt(0).sum()) if len(g) else 0,
                    int(g.drop_persists_frozen_alignment.sum()) if len(g) else 0,
                    int(g.drop_persists_frozen_alignment_and_matching.sum()) if len(g) else 0,
                ]
            )
        table(
            "registration_audit",
            "Sensitivity of decreasing adjacent-decile transitions to registration and equivalent-part correspondence. Columns count all PA drops, transitions with a changed assignment, drops that persist with the preceding global alignment fixed, and drops that persist with both preceding alignment and matching fixed. These are transition counts, not evaluation counts or causal error labels.",
            "tab:registration_audit",
            ["System", "PA drops", "Assignment changed", "Fixed alignment", "Both fixed"],
            audit_rows,
        )
    if (pa.CACHE / "quality_declines.json").exists():
        decline = pd.DataFrame(pa.read_json(pa.CACHE / "quality_declines.json"))
        checks = []
        for s, g in decline.groupby("system", sort=False):
            checks.append(
                [
                    pa.SYSTEMS[s],
                    len(g),
                    int(g["drop"].sum()),
                    int(g.recovered.sum()),
                    f"{100 * pa.overall(g, 'loss'):.2f}",
                ]
            )
        table(
            "trajectory_audit",
            "Offline trajectory coverage, PA decreases between adjacent sampled checkpoints, recovery to the observed peak by termination, and source-weighted mean peak-to-final PA loss (percentage points). Counts refer to evaluations; PartNet contributes two conditions per shape.",
            "tab:trajectory_audit",
            ["System", "Evaluated", "Any drop", "Recovered", "Mean loss"],
            checks,
        )
    else:
        (pa.PAPER / "tab/trajectory_audit.tex").write_text(
            r"\TODO{Complete and validate recorded-state quality scoring before reporting regression counts.}"
            + "\n"
        )
