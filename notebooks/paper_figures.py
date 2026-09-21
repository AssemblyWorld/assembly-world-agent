"""PDF figures derived from recorded episodes and notebook-owned statistics."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import paper_analysis as pa
import paper_geometry as pg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from assembly_world_agent.episode_io import read_episode
from assembly_world_agent.utils import rotation_matrix
from assembly_world_agent.vis.benchmark import decimate_mesh
from assembly_world_agent.vis.results import episode_data, prompt_manual_pages


def load_analyses(frame):
    analyses, absent = [], []
    for row in frame.to_dict("records"):
        p = (
            pa.CACHE
            / "trajectories"
            / row["system"]
            / row["block"]
            / f"{row['sample_id'].replace('/', '--')}.json"
        )
        if not p.exists():
            absent.append((row["system"], row["block"], row["sample_id"]))
            continue
        d = pa.read_json(p)
        _, states, calls, sha = pg.recorded_episode(row)
        checkpoints, _ = pg.sample_checkpoints(row, states, calls)
        input_sha = pa.digest(pg.evaluation_cache_path(pa.ROOT / row["input"]["initial_path"]))
        if d["key"] != pg.analysis_key(row, sha, input_sha, checkpoints):
            absent.append((row["system"], row["block"], row["sample_id"], "stale analysis key"))
            continue
        if d["scores"]:
            pg.state_cache_protocol()
        analyses.append(d)
    return analyses, absent


def geometry(row):
    values, _ = pg.evaluator_inputs(row)
    ep = read_episode(Path(row["sample_dir"]) / "final.episode.zip")
    data = episode_data(ep)
    meshes = []
    for part in data["parts"]:
        g = part["geometry"]
        v, t, _ = decimate_mesh(
            np.asarray(g["vertices"]), np.asarray(g["triangles"]).reshape(-1, 3), limit=4000
        )
        meshes.append((v, t))
    return values, meshes, data


def transformed(meshes, poses, divisor, alignment=None):
    out = []
    for (v, t), q in zip(meshes, poses):
        v = (v @ rotation_matrix(q[3:]).T + q[:3]) / divisor
        if alignment:
            v = v @ np.asarray(alignment["rotation"]).T + alignment["translation"]
        out.append((v, t))
    return out


def draw_mesh(ax, meshes, colors=None, view=(24, -55), limits=None):
    all_v = np.concatenate([v for v, t in meshes])
    low, high = all_v.min(0), all_v.max(0)
    center = (low + high) / 2
    radius = max((high - low).max() / 2, 0.01) * 1.06
    if limits is not None:
        center, radius = limits
    for i, (v, t) in enumerate(meshes):
        color = colors[i] if colors is not None else pa.plt.get_cmap("tab20")(i % 20)
        p = Poly3DCollection(
            v[t],
            facecolors=color,
            edgecolor="none",
            linewidth=0,
            shade=True,
            zsort="average",
            rasterized=True,
        )
        ax.add_collection3d(p)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(*view)
    ax.set_axis_off()
    ax.set_proj_type("ortho")
    return (center, radius)


def sample_row(frame, system, block, sid):
    return (
        frame[(frame.system == system) & (frame.block == block) & (frame.sample_id == sid)]
        .iloc[0]
        .to_dict()
    )


def make_cross_domain(frame):
    import base64

    selections = []
    fig = pa.plt.figure(figsize=(7, 5.2), layout="constrained")
    grid = fig.add_gridspec(4, 4)
    for k, b in enumerate(pa.BLOCKS[1:]):
        f = frame[(frame.system == "gpt-6-astra") & (frame.block == b)].copy()
        f["deviation"] = (f.PA - f.PA.median()).abs()
        f["part_deviation"] = (f.parts - f.parts.median()).abs()
        row = f.sort_values(["deviation", "part_deviation", "sample_id"]).iloc[0].to_dict()
        values, meshes, data = geometry(row)
        final = data["frames"][str(row["metric"]["state_index"])]["poses"]
        initial = data["frames"]["0"]["poses"]
        gt = [np.r_[g.position, g.quaternion] for g in values["gt_poses"]]
        pred = transformed(meshes, final, values["divisor"], row["metric"]["alignment"])
        target = transformed(meshes, gt, values["divisor"])
        for col, m in [
            (0, transformed(meshes, initial, values["divisor"])),
            (2, pred),
            (3, target),
        ]:
            ax = fig.add_subplot(grid[k, col], projection="3d")
            draw_mesh(ax, m)
            if k == 0:
                ax.set_title(["Initial state", "Reference", "Agent result", "Target"][col])
            if col == 0:
                ax.text2D(
                    0,
                    -0.04,
                    f"{pa.BLOCK_LABELS[k + 1]}\n{row['sample_id']} / {row['parts']} parts",
                    transform=ax.transAxes,
                    fontsize=9,
                )
            if col == 2:
                ax.text2D(
                    0.05,
                    -0.04,
                    f"PA {row['PA']:.2f} / SR {row['SR']}",
                    transform=ax.transAxes,
                    fontsize=9,
                )
        ax = fig.add_subplot(grid[k, 1])
        ax.axis("off")
        if k == 0:
            ax.set_title("Provided reference")
        errors = []
        pages = (
            prompt_manual_pages(Path(row["sample_dir"]), errors)
            if b != "fantastic-breaks-none"
            else []
        )
        if pages:
            raw = base64.b64decode(pages[-1]["data"].split(",", 1)[1])
            ax.imshow(Image.open(io.BytesIO(raw)))
            ax.text(
                0.5,
                -0.04,
                f"Page {len(pages)} / {len(pages)}",
                transform=ax.transAxes,
                ha="center",
                fontsize=9,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No reference" if b == "fantastic-breaks-none" else "Reference not decoded",
                ha="center",
                va="center",
                fontsize=9,
            )
        selections.append(
            dict(
                block=b,
                sample_id=row["sample_id"],
                PA=row["PA"],
                SR=row["SR"],
                rule="minimum absolute deviation from block median PA; then distance to median part count; then sample id",
                reference_errors=errors,
                episode_sha256=row["metric"]["episode_sha256"],
            )
        )
    pa.save_json(pa.CACHE / "qualitative_selection.json", selections)
    return pa.save_figure(fig, "cross_domain")


def make_system_comparison(frame):
    selections = pa.read_json(pa.CACHE / "qualitative_selection.json")
    outputs = []
    for half in range(2):
        systems = list(pa.SYSTEMS)[half * 4 : (half + 1) * 4]
        fig = pa.plt.figure(figsize=(7, 5.6), layout="constrained")
        gs = fig.add_gridspec(4, 5)
        for i, selection in enumerate(selections):
            base = sample_row(frame, "gpt-6-astra", selection["block"], selection["sample_id"])
            values, meshes, _ = geometry(base)
            gt = [np.r_[g.position, g.quaternion] for g in values["gt_poses"]]
            for j, system in enumerate(systems):
                row = sample_row(frame, system, selection["block"], selection["sample_id"])
                _, states, _, _ = pg.recorded_episode(row)
                q = states[row["metric"]["state_index"]]
                ax = fig.add_subplot(gs[i, j], projection="3d")
                draw_mesh(ax, transformed(meshes, q, values["divisor"], row["metric"]["alignment"]))
                if i == 0:
                    ax.set_title(pa.SHORT_SYSTEMS[system], fontsize=9)
                ax.text2D(0.08, -0.08, f"PA {row['PA']:.2f}", transform=ax.transAxes, fontsize=9)
            ax = fig.add_subplot(gs[i, 4], projection="3d")
            draw_mesh(ax, transformed(meshes, gt, values["divisor"]))
            if i == 0:
                ax.set_title("Target", fontsize=9)
            ax.text2D(0.02, -0.08, selection["sample_id"], transform=ax.transAxes, fontsize=9)
        outputs.append(pa.save_figure(fig, "system_comparison", f"system_comparison_{half + 1}"))
    return outputs


def make_failure_analysis(frame, analyses):
    if len(analyses) != len(frame):
        raise ValueError(
            "Failure analysis requires validated diagnostics for every benchmark evaluation"
        )
    categories = [
        "Translation dominated",
        "Orientation dominated",
        "Mixed",
        "Near threshold",
        "Far displaced",
        "Never moved",
    ]
    palette = ["#e78c39", "#729ac0", "#8467a1", "#5daa8e", "#cf6057", "#b4b4b4"]
    records = []
    for d in analyses:
        for part in d["diagnostics"]:
            records.append(
                dict(system=d["system"], block=d["block"], sample_id=d["sample_id"], **part)
            )
    parts = pd.DataFrame(records)
    errors = parts[parts.category != "Correct"]
    rows = []
    for s in pa.SYSTEMS:
        f = errors[errors.system == s]
        count = f.category.value_counts()
        rows.append(
            dict(
                system=s,
                wrong_parts=len(f),
                failed_shapes=len(f.groupby(["block", "sample_id"])),
                **{c: float(count.get(c, 0) / len(f)) if len(f) else 0 for c in categories},
            )
        )
    pa.save_json(pa.CACHE / "failure_summary.json", rows)
    parts.to_csv(pa.CACHE / "part_diagnostics.csv", index=False)
    fig, ax = pa.plt.subplots(figsize=(7, 2.6), layout="constrained")
    bottom = np.zeros(8)
    for cat, color in zip(categories, palette):
        vals = np.array([r[cat] for r in rows])
        ax.barh(range(8), vals, left=bottom, color=color, label=cat.replace(" dominated", ""))
        bottom += vals
    ax.set_yticks(range(8), pa.SYSTEMS.values())
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of incorrectly placed parts")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=9)
    pa.save_figure(fig, "failures", "failure_distribution")
    # Shape-equal and source-specific diagnostics prevent large objects dominating silently.
    shape = (
        errors.groupby(["system", "block", "sample_id"])
        .category.value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    shape.groupby("system").mean().to_csv(pa.CACHE / "failure_shape_equal.csv")
    errors.groupby(["system", "block"]).category.value_counts(normalize=True).to_csv(
        pa.CACHE / "failure_by_source.csv"
    )
    sensitivity = []
    for system in pa.SYSTEMS:
        e = errors[errors.system == system]
        eligible = ~e.category.isin(["Never moved", "Far displaced"])
        for distance in [0.05, 0.1, 0.2]:
            for centered in [0.005, 0.01, 0.02]:
                fraction = (
                    eligible & (e.centroid_distance > distance) & (e.centered_CD <= centered)
                ).mean()
                sensitivity.append(
                    dict(
                        system=system,
                        centroid_threshold=distance,
                        centered_cd_threshold=centered,
                        translation_fraction=float(fraction),
                    )
                )
    pa.save_json(pa.CACHE / "failure_sensitivity.json", sensitivity)
    # Choose moderate translation errors deterministically, not the most extreme ones.
    candidates = errors[
        (errors.system == "gpt-6-astra") & (errors.category == "Translation dominated")
    ]
    selected = []
    for block in ["ikea-manualbook", "partnet-final-image"]:
        f = candidates[candidates.block == block].copy()
        if f.empty:
            continue
        f["deviation"] = (f.centroid_distance - f.centroid_distance.median()).abs()
        selected.append(f.sort_values(["deviation", "sample_id", "part_id"]).iloc[0].to_dict())
    fig = pa.plt.figure(figsize=(7, 2.6), layout="constrained")
    gs = fig.add_gridspec(1, len(selected))
    for k, p in enumerate(selected):
        row = sample_row(frame, p["system"], p["block"], p["sample_id"])
        values, meshes, data = geometry(row)
        ids = values["part_ids"]
        q = data["frames"][str(row["metric"]["state_index"])]["poses"]
        gt = [np.r_[g.position, g.quaternion] for g in values["gt_poses"]]
        target = transformed(meshes, gt, values["divisor"])
        pred = transformed(meshes, q, values["divisor"], row["metric"]["alignment"])
        i, j = ids.index(p["part_id"]), ids.index(p["target_part_id"])
        draw = target + [pred[i]]
        colors = [(0.72, 0.72, 0.72, 0.22)] * len(target)
        colors[j] = "#29916c"
        colors += ["#e78c39"]
        ax = fig.add_subplot(gs[0, k], projection="3d")
        draw_mesh(ax, draw, colors)
        ax.text2D(
            0.02,
            0,
            f"{p['sample_id']} / part {p['part_id']}\ncentroid offset {p['centroid_distance']:.2f}; centered CD {p['centered_CD']:.1e}",
            transform=ax.transAxes,
            fontsize=9,
        )
    pa.save_figure(fig, "failures", "failure_examples")
    fig = pa.plt.figure(figsize=(7, 2.8), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[3.6, 1.7, 1.7])
    ax = fig.add_subplot(gs[0, 0])
    bottom = np.zeros(8)
    for cat, color in zip(categories, palette):
        vals = np.array([r[cat] for r in rows])
        ax.barh(range(8), vals, left=bottom, color=color, label=cat.replace(" dominated", ""))
        bottom += vals
    ax.set_yticks(range(8), [pa.SHORT_SYSTEMS[s] for s in pa.SYSTEMS], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of incorrect parts", fontsize=9)
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(-0.05, 1.01), fontsize=9)
    for k, p in enumerate(selected):
        row = sample_row(frame, p["system"], p["block"], p["sample_id"])
        values, meshes, data = geometry(row)
        ids = values["part_ids"]
        q = data["frames"][str(row["metric"]["state_index"])]["poses"]
        target = transformed(
            meshes, [np.r_[g.position, g.quaternion] for g in values["gt_poses"]], values["divisor"]
        )
        pred = transformed(meshes, q, values["divisor"], row["metric"]["alignment"])
        i, j = ids.index(p["part_id"]), ids.index(p["target_part_id"])
        colors = [(0.72, 0.72, 0.72, 0.22)] * len(target)
        colors[j] = "#29916c"
        ax = fig.add_subplot(gs[0, k + 1], projection="3d")
        draw_mesh(ax, target + [pred[i]], colors + ["#e78c39"])
        ax.set_title(p["sample_id"], fontsize=9)
        ax.text2D(
            0.02,
            -0.04,
            f"Part {p['part_id']}\noffset {p['centroid_distance']:.2f}",
            transform=ax.transAxes,
            fontsize=9,
        )
    pa.save_figure(fig, "failures", "failures_main")
    pa.save_json(pa.CACHE / "failure_examples.json", selected)
    return rows


def make_behavior(frame, analyses):
    if len(analyses) != len(frame):
        raise ValueError("Behavior analysis requires all benchmark episode records")
    actions = []
    for d in analyses:
        for c in d["actions"]:
            actions.append(
                dict(system=d["system"], block=d["block"], sample_id=d["sample_id"], **c)
            )
    a = pd.DataFrame(actions)
    category = {
        "capture_scene": "Observe",
        "move_camera": "Observe",
        "get_object": "Inspect",
        "get_scene": "Inspect",
        "get_state": "Inspect",
        "list_objects": "Inspect",
        **{k: "Manipulate" for k in pg.EDIT_TOOLS},
        **{k: "Manage" for k in ["start_episode", "group_objects", "ungroup_objects"]},
    }
    a["category"] = a.name.map(category)
    edges = np.linspace(0, 1, 41)
    x = (edges[:-1] + edges[1:]) / 2
    selections = [
        (pa.REPRESENTATIVES, "behavior"),
        (list(pa.SYSTEMS)[:4], "behavior_all_1"),
        (list(pa.SYSTEMS)[4:], "behavior_all_2"),
    ]
    for selection, name in selections:
        fig, axes = pa.plt.subplots(
            len(selection), 3, figsize=(7, 1.35 * len(selection)), layout="constrained"
        )
        for i, s in enumerate(selection):
            f = a[a.system == s]
            for cat, color in zip(
                ["Inspect", "Manipulate", "Observe", "Manage"],
                ["#777777", "#e78c39", "#1769aa", "#6c9c75"],
            ):
                selected = f[f.category == cat]
                weights = selected.block.map(pa.WEIGHTS).to_numpy() / 20
                hist = np.histogram(selected.fraction, edges, weights=weights)[0] / 0.025
                axes[i, 0].plot(x, gaussian_filter1d(hist, 1.2), label=cat, color=color, lw=1)
            axes[i, 0].set_ylabel(pa.SHORT_SYSTEMS[s] + "\nCalls / time", fontsize=9)
            edits = f[
                f.name.isin(pg.EDIT_TOOLS) & ((f.displacement > 1e-9) | (f.angle >= 0.5))
            ].copy()
            edits["bin"] = np.minimum((edits.fraction * 10).astype(int), 9)
            for j, column, label in [
                (1, "displacement", "Centroid displacement"),
                (2, "angle", "Rotation (degrees)"),
            ]:
                z = edits[edits.angle < 0.5] if j == 1 else edits[edits.angle >= 0.5]
                g = z.groupby("bin")[column]
                med = g.median()
                lo = g.quantile(0.25)
                hi = g.quantile(0.75)
                axes[i, j].plot((med.index + 0.5) / 10, med, color=pa.COLORS[s], lw=1.3)
                axes[i, j].fill_between(
                    (med.index + 0.5) / 10, lo, hi, color=pa.COLORS[s], alpha=0.18
                )
                if j == 1:
                    axes[i, j].set_yscale("log")
                    axes[i, j].set_ylim(0.001, 10)
                else:
                    axes[i, j].set_ylim(0, 190)
                if i == 0:
                    axes[i, j].set_title(label, fontsize=9)
            for ax in axes[i]:
                ax.set_xlim(0, 1)
                ax.grid(alpha=0.15)
                if i == len(selection) - 1:
                    ax.set_xlabel("Episode fraction")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, fontsize=9, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01)
        )
        pa.save_figure(fig, "behavior" if name == "behavior" else "appendix_behavior", name)
    summaries = []
    for s in pa.SYSTEMS:
        f = a[a.system == s]
        edits = f[f.name.isin(pg.EDIT_TOOLS)].dropna(subset=["primary"])
        summaries.append(
            dict(
                system=s,
                calls=len(f),
                capture=int((f.name == "capture_scene").sum()),
                edits=len(edits),
                median_displacement=float(edits.displacement.median()),
                median_rotation=float(edits.loc[edits.angle >= 0.5, "angle"].median()),
            )
        )
    pa.save_json(pa.CACHE / "behavior_summary.json", summaries)
    return a


def make_quality(frame, analyses):
    def clear_unvalidated_outputs():
        for relative in [
            "behavior/quality.pdf",
            "appendix_behavior/quality_all.pdf",
            "appendix_behavior/budget_all.pdf",
        ]:
            (pa.PAPER / "fig" / relative).unlink(missing_ok=True)
        for name in ["quality_curves.csv", "quality_checkpoints.csv", "quality_declines.json"]:
            (pa.CACHE / name).unlink(missing_ok=True)

    checkpoints = []
    for d in analyses:
        if not d["scores"]:
            continue
        for kind, t, si in d["key"]["checkpoints"]:
            checkpoints.append(
                dict(
                    system=d["system"],
                    block=d["block"],
                    sample_id=d["sample_id"],
                    kind=kind,
                    time=t,
                    **{k: d["scores"][str(si)][k] for k in ["PA", "SR", "SCD"]},
                )
            )
    f = pd.DataFrame(checkpoints)
    if f.empty:
        clear_unvalidated_outputs()
        return None
    # Never draw a system-level curve from an incomplete analysis silently.
    complete = [
        s
        for s in pa.SYSTEMS
        if len(f[(f.system == s) & (f.kind == "fraction") & (f.time == 1)]) == 100
    ]
    if not complete:
        clear_unvalidated_outputs()
        return None
    for system in complete:
        final = f[(f.system == system) & (f.kind == "fraction") & (f.time == 1)]
        official = frame[frame.system == system]
        for metric in ["PA", "SR"]:
            if not np.isclose(pa.overall(final, metric), pa.overall(official, metric), atol=1e-8):
                raise ValueError(f"{system}: {metric} curve endpoint differs from the result table")
    summary = []
    for (s, kind, t), g in f[f.system.isin(complete)].groupby(["system", "kind", "time"]):
        summary.append(
            dict(
                system=s,
                kind=kind,
                time=t,
                PA=pa.overall(g, "PA"),
                SR=pa.overall(g, "SR"),
                SCD_median=pa.source_weighted_median(g, "SCD"),
                PA_ci=pa.bootstrap(pa.shape_values(g, "PA")),
            )
        )
    pd.DataFrame(summary).to_csv(pa.CACHE / "quality_curves.csv", index=False)
    f.to_csv(pa.CACHE / "quality_checkpoints.csv", index=False)
    sframe = pd.DataFrame(summary)
    if sframe.empty:
        return sframe
    for selection, name in [(pa.REPRESENTATIVES, "quality"), (list(pa.SYSTEMS), "quality_all")]:
        fig, axes = pa.plt.subplots(1, 3, figsize=(7, 2.1), layout="constrained")
        for s in selection:
            g = sframe[(sframe.system == s) & (sframe.kind == "fraction")]
            if g.empty:
                continue
            for ax, metric in zip(axes, ["PA", "SR", "SCD_median"]):
                ax.plot(
                    g.time,
                    g[metric] * (1000 if metric == "SCD_median" else 1),
                    label=pa.SHORT_SYSTEMS[s],
                    color=pa.COLORS[s],
                    lw=1.4,
                )
                if metric == "PA":
                    ax.fill_between(
                        g.time,
                        [v[0] for v in g.PA_ci],
                        [v[1] for v in g.PA_ci],
                        color=pa.COLORS[s],
                        alpha=0.10,
                    )
                ax.set_xlabel("Episode fraction")
                ax.set_ylabel("Median SCD (×1000)" if metric == "SCD_median" else metric)
                ax.grid(alpha=0.2)
        axes[0].set_ylim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[2].set_yscale("log")
        axes[0].legend(fontsize=9)
        pa.save_figure(fig, "behavior" if name == "quality" else "appendix_behavior", name)
    fig, axes = pa.plt.subplots(1, 2, figsize=(7, 2.5), layout="constrained")
    terminal_audit = []
    for s in complete:
        g = sframe[(sframe.system == s) & (sframe.kind == "budget")]
        terminal = sframe[
            (sframe.system == s) & (sframe.kind == "fraction") & (sframe.time == 1)
        ].iloc[0]
        budget60 = g[g.time == 60].iloc[0]
        terminal_audit.append(
            dict(
                system=s,
                PA_at_60=float(budget60.PA),
                PA_terminal=float(terminal.PA),
                SR_at_60=float(budget60.SR),
                SR_terminal=float(terminal.SR),
            )
        )
        if s in pa.REPRESENTATIVES and not np.allclose(
            [budget60.PA, budget60.SR], [terminal.PA, terminal.SR], atol=1e-9
        ):
            raise ValueError(
                "A representative curve no longer reaches its table value at 60 minutes"
            )
        for ax, metric in zip(axes, ["PA", "SR"]):
            ax.plot(
                g.time, g[metric], "o-", label=pa.SHORT_SYSTEMS[s], color=pa.COLORS[s], ms=3, lw=1
            )
            ax.set_xscale("log")
            ax.scatter(100, terminal[metric], marker="D", color=pa.COLORS[s], s=15)
            ax.set_xticks([1, 2, 5, 10, 20, 30, 60, 100], [1, 2, 5, 10, 20, 30, 60, "Final"])
            ax.set_xlabel("Absolute truncation time (minutes)")
            ax.set_ylabel("Overall " + metric)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.2)
    for ax in axes:
        ax.axvspan(80, 125, color="0.95", zorder=-1)
        ax.axvline(80, color="0.7", lw=0.5, ls=":")
    pa.save_json(pa.CACHE / "budget_terminal_audit.json", terminal_audit)
    axes[0].legend(ncol=2, fontsize=9)
    pa.save_figure(fig, "appendix_behavior", "budget_all")
    decline = []
    for (s, b, sid), g in f[(f.kind == "fraction") & f.system.isin(complete)].groupby(
        ["system", "block", "sample_id"]
    ):
        g = g.sort_values("time")
        v = g.PA.to_numpy()
        drops = np.diff(v) < -1e-9
        decline.append(
            dict(
                system=s,
                block=b,
                sample_id=sid,
                drop=bool(drops.any()),
                recovered=bool(drops.any() and v[-1] >= v.max() - 1e-9),
                loss=float(v.max() - v[-1]),
            )
        )
    pa.save_json(pa.CACHE / "quality_declines.json", decline)
    return sframe


def make_efficiency(rows, quality=None):
    available = set(quality.system) if quality is not None and not quality.empty else set()
    n = 3 if set(pa.REPRESENTATIVES) <= available else 2
    fig, axes = pa.plt.subplots(1, n, figsize=(7, 2.5), layout="constrained")
    if n == 2:
        fig.suptitle("Retrospective budget curves pending complete scoring", fontsize=9)
    for row in rows:
        s = row["system"]
        for ax, key, xlabel in zip(
            axes[:2],
            ["mean_minutes", "cost"],
            ["Mean wall time (minutes)", "Mean cost (USD)"],
        ):
            hollow = key == "cost" and row["cost_n"] < 100
            ax.scatter(
                row[key],
                row["SR"],
                s=26,
                edgecolor=pa.COLORS[s],
                facecolor="white" if hollow else pa.COLORS[s],
            )
            offset = (3, 4 if s != "gpt-5.6-terra" else -9)
            horizontal = "left"
            if key == "cost":
                offset = {
                    "qwen3.8-max-litellm": (-3, 32),
                    "gpt-5.6-sol": (9, 0),
                    "gpt-5.6-terra": (1, 15),
                    "claude-sonnet-5": (12, 9),
                    "deepseek-v4.1-flash": (4, 4),
                }.get(s, offset)
            ax.annotate(
                pa.SHORT_SYSTEMS[s],
                (row[key], row["SR"]),
                xytext=offset,
                ha=horizontal,
                textcoords="offset points",
                fontsize=9,
                arrowprops=dict(arrowstyle="-", color="0.55", lw=0.5, shrinkA=2, shrinkB=4)
                if key == "cost"
                and s in {"qwen3.8-max-litellm", "gpt-5.6-terra", "claude-sonnet-5", "gpt-5.6-sol"}
                else None,
            )
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("Overall SR")
            ax.set_ylim(-0.02, 0.72)
            ax.margins(x=0.18)
            ax.grid(alpha=0.18)
    if n == 3:
        for s in pa.REPRESENTATIVES:
            g = quality[(quality.system == s) & (quality.kind == "budget")]
            axes[2].plot(
                g.time, g.SR, "o-", color=pa.COLORS[s], label=pa.SHORT_SYSTEMS[s], ms=2, lw=1
            )
        axes[2].set_xscale("log")
        axes[2].set_xticks([1, 5, 10, 30, 60], [1, 5, 10, 30, 60])
        axes[2].set_xlabel("Truncation time (minutes)")
        axes[2].set_ylabel("Overall SR")
        axes[2].set_ylim(-0.02, 0.72)
        axes[2].legend(fontsize=9, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
        axes[2].grid(alpha=0.18)
    pa.save_figure(fig, "efficiency")


def make_reference_examples(frame):
    base = frame[(frame.system == "gpt-6-astra") & frame.block.str.startswith("partnet")]
    p = base.pivot(index="sample_id", columns="block", values="PA")
    p["delta"] = p[pa.BLOCKS[1]] - p[pa.BLOCKS[0]]
    candidates = p[p.delta > 0].copy()
    candidates["deviation"] = (candidates.delta - candidates.delta.median()).abs()
    sid = candidates.sort_values(["deviation", "sample_id"]).index[0]
    base_row = sample_row(frame, "gpt-6-astra", pa.BLOCKS[0], sid)
    values, meshes, _ = geometry(base_row)
    fig = pa.plt.figure(figsize=(7, 2.0), layout="constrained")
    gs = fig.add_gridspec(1, 3)
    for j, b in enumerate(pa.BLOCKS[:2]):
        row = sample_row(frame, "gpt-6-astra", b, sid)
        _, states, _, _ = pg.recorded_episode(row)
        ax = fig.add_subplot(gs[0, j], projection="3d")
        draw_mesh(
            ax,
            transformed(
                meshes,
                states[row["metric"]["state_index"]],
                values["divisor"],
                row["metric"]["alignment"],
            ),
        )
        ax.set_title(
            f"{'No reference' if j == 0 else 'Final image'}: PA {row['PA']:.2f}", fontsize=9
        )
    gt = [np.r_[g.position, g.quaternion] for g in values["gt_poses"]]
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    draw_mesh(ax, transformed(meshes, gt, values["divisor"]))
    ax.set_title("Target", fontsize=9)
    pa.save_figure(fig, "reference_analysis", "reference_example")
    fig = pa.plt.figure(figsize=(7, 2.4), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[2.8, 1.4, 1.4, 1.4])
    ax = fig.add_subplot(gs[0, 0])
    for i, system in enumerate(pa.SYSTEMS):
        paired = (
            frame[(frame.system == system) & frame.block.isin(pa.BLOCKS[:2])]
            .pivot(index="sample_id", columns="block", values="PA")
            .sort_index()
        )
        differences = (paired[pa.BLOCKS[1]] - paired[pa.BLOCKS[0]]).to_numpy()
        r = dict(delta=float(differences.mean()), ci=pa.bootstrap(differences[None, :]))
        ax.errorbar(
            r["delta"],
            i,
            xerr=[[r["delta"] - r["ci"][0]], [r["ci"][1] - r["delta"]]],
            fmt="o",
            color=pa.COLORS[system],
            capsize=2,
            ms=3,
        )
    ax.set_yticks(range(8), [pa.SHORT_SYSTEMS[s] for s in pa.SYSTEMS], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color=".6", lw=0.7)
    ax.set_xlabel("Paired PA gain (image - none)", fontsize=9)
    for j, b in enumerate(pa.BLOCKS[:2]):
        row = sample_row(frame, "gpt-6-astra", b, sid)
        _, states, _, _ = pg.recorded_episode(row)
        ax = fig.add_subplot(gs[0, j + 1], projection="3d")
        draw_mesh(
            ax,
            transformed(
                meshes,
                states[row["metric"]["state_index"]],
                values["divisor"],
                row["metric"]["alignment"],
            ),
        )
        ax.set_title(
            f"{'No reference' if j == 0 else 'Final image'}\nPA {row['PA']:.2f}", fontsize=9
        )
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    draw_mesh(ax, transformed(meshes, gt, values["divisor"]))
    ax.set_title("Target", fontsize=9)
    pa.save_figure(fig, "reference_analysis", "reference_main")
    pa.save_json(
        pa.CACHE / "reference_example.json",
        dict(
            sample_id=sid,
            rule="median positive Astra PA gain; nearest, then sample id",
            none=float(p.loc[sid, pa.BLOCKS[0]]),
            image=float(p.loc[sid, pa.BLOCKS[1]]),
        ),
    )


def make_trajectory(frame):
    f = frame[
        (frame.system == "gpt-6-astra") & (frame.block == "ikea-manualbook") & (frame.SR == 1)
    ].copy()
    f["deviation"] = (f.duration - f.duration.median()).abs()
    row = f.sort_values(["deviation", "sample_id"]).iloc[0].to_dict()
    values, meshes, _ = geometry(row)
    _, states, calls, _ = pg.recorded_episode(row)
    cps, fractions = pg.sample_checkpoints(row, states, calls)
    fig = pa.plt.figure(figsize=(7, 1.9), layout="constrained")
    gs = fig.add_gridspec(1, 4)
    chosen = []
    for j, k in enumerate([0, 3, 7, 10]):
        _, frac, si = cps[k]
        ax = fig.add_subplot(gs[0, j], projection="3d")
        draw_mesh(
            ax, transformed(meshes, states[si], values["divisor"], row["metric"]["alignment"])
        )
        reached = [c for c, f in zip(calls, fractions) if f <= frac + 1e-12]
        label = reached[-1]["name"] if reached else "initial state"
        ax.set_title(f"t = {frac:.1f}", fontsize=9)
        ax.text2D(0.0, -0.03, label.replace("_", " "), transform=ax.transAxes, fontsize=9)
        chosen.append(dict(fraction=frac, state_index=si, last_call=label))
    pa.save_figure(fig, "qualitative")
    pa.save_json(
        pa.CACHE / "trajectory_example.json",
        dict(
            sample_id=row["sample_id"],
            block=row["block"],
            rule="successful Astra IKEA episode nearest median successful duration",
            states=chosen,
        ),
    )


def make_strategy(frame, analyses):
    rows = []
    for d in analyses:
        edits = [x for x in d["actions"] if x["name"] in pg.EDIT_TOOLS and x.get("moved")]
        visited = set()
        last = {}
        revisits = reverse = pairs = 0
        for x in edits:
            revisits += bool(set(x["moved"]) & visited)
            visited.update(x["moved"])
            if x["angle"] >= 0.5:
                continue
            if "vectors" not in x:
                raise ValueError("Refresh trajectory analyses to include per-part motion vectors")
            for pid, vector in x["vectors"].items():
                v = np.asarray(vector)
                if np.linalg.norm(v) <= 1e-9:
                    continue
                if pid in last:
                    reverse += np.dot(v, last[pid]) < 0
                    pairs += 1
                last[pid] = v
        captures = [x for x in d["actions"] if x["name"] == "capture_scene"]
        distances = [
            np.linalg.norm(np.asarray(x["camera"]["position"]) - x["camera"]["target"])
            for x in captures
            if x.get("camera")
        ]
        rows.append(
            dict(
                system=d["system"],
                block=d["block"],
                sample_id=d["sample_id"],
                edit_count=len(edits),
                capture_count=len(captures),
                revisit_fraction=revisits / len(edits) if edits else np.nan,
                reversal_fraction=reverse / pairs if pairs else np.nan,
                captures_per_edit=len(captures) / len(edits) if edits else np.nan,
                final_camera_ratio=distances[-1] / distances[0]
                if distances and distances[0] > 0
                else np.nan,
            )
        )
    s = pd.DataFrame(rows)
    s.to_csv(pa.CACHE / "strategy.csv", index=False)
    fig, axes = pa.plt.subplots(2, 2, figsize=(7, 4), layout="constrained")
    for ax, col, label in zip(
        axes.flat,
        ["revisit_fraction", "reversal_fraction", "captures_per_edit", "final_camera_ratio"],
        [
            "Edits to previously edited parts",
            "Opposed consecutive translations",
            "Captures per edit",
            "Final / first camera distance",
        ],
    ):
        for i, system in enumerate(pa.SYSTEMS):
            vals = s[s.system == system][col].dropna()
            if len(vals):
                med = vals.median()
                lo, hi = vals.quantile([0.25, 0.75])
                ax.errorbar(
                    i,
                    med,
                    yerr=[[med - lo], [hi - med]],
                    fmt="o",
                    color=pa.COLORS[system],
                    capsize=2,
                    ms=3,
                )
        ax.set_xticks(
            range(8),
            list(pa.SHORT_SYSTEMS.values()),
            rotation=40,
            ha="right",
            fontsize=9,
        )
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
    pa.save_figure(fig, "appendix_behavior", "strategy")
    return s


def make_main_behavior(analyses, quality=None):
    """Compact question-led main-text panel; full action plots stay in the appendix."""
    from matplotlib.patches import Patch

    fig, axes = pa.plt.subplots(1, 3, figsize=(7, 2.6), layout="constrained")
    categories = ["Inspect", "Manipulate", "Observe", "Manage"]
    category_colors = ["#969696", "#e78c39", "#1769aa", "#6c9c75"]
    mapping = {
        "get_object": "Inspect",
        "get_scene": "Inspect",
        "get_state": "Inspect",
        "list_objects": "Inspect",
        "capture_scene": "Observe",
        "move_camera": "Observe",
        **{k: "Manipulate" for k in pg.EDIT_TOOLS},
        **{k: "Manage" for k in ["start_episode", "group_objects", "ungroup_objects"]},
    }
    for i, system in enumerate(pa.REPRESENTATIVES):
        actions = [
            dict(**a, block=d["block"])
            for d in analyses
            if d["system"] == system
            for a in d["actions"]
        ]
        for k in range(10):
            selected = [a for a in actions if min(int(a["fraction"] * 10), 9) == k]
            counts = np.array(
                [
                    sum(
                        pa.WEIGHTS[a["block"]] / 20 for a in selected if mapping.get(a["name"]) == c
                    )
                    for c in categories
                ]
            )
            share = counts / counts.sum() if counts.sum() else np.zeros_like(counts)
            bottom = float(i)
            for value, color in zip(share, category_colors):
                axes[0].bar((k + 0.5) / 10, value * 0.75, width=0.095, bottom=bottom, color=color)
                bottom += value * 0.75
        edits = pd.DataFrame(
            [a for a in actions if a.get("displacement", 0) > 1e-9 and a.get("angle", 0) < 0.5]
        )
        if not edits.empty:
            edits["bin"] = np.minimum((edits.fraction * 10).astype(int), 9)
            grouped = edits.groupby("bin").displacement
            med, lo, hi = grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75)
            x = (med.index + 0.5) / 10
            axes[1].plot(x, med, color=pa.COLORS[system], label=pa.SHORT_SYSTEMS[system], lw=1.2)
            axes[1].fill_between(x, lo, hi, color=pa.COLORS[system], alpha=0.10)
        if quality is not None and not quality.empty:
            g = quality[(quality.system == system) & (quality.kind == "fraction")]
            axes[2].plot(g.time, g.PA, color=pa.COLORS[system], lw=1.2)
            if not g.empty:
                axes[2].fill_between(
                    g.time,
                    [c[0] for c in g.PA_ci],
                    [c[1] for c in g.PA_ci],
                    color=pa.COLORS[system],
                    alpha=0.10,
                )
    axes[0].set_yticks(
        np.arange(4) + 0.375,
        [pa.SHORT_SYSTEMS[s] for s in pa.REPRESENTATIVES],
        fontsize=9,
    )
    axes[0].set_ylim(3.9, -0.1)
    axes[0].legend(
        handles=[
            Patch(color=color, label=label) for color, label in zip(category_colors, categories)
        ],
        ncol=2,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0, -0.25),
    )
    axes[0].set_title("(a) Call composition", fontsize=9)
    axes[1].set_title("(b) Translation magnitude", fontsize=9)
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.001, 10)
    axes[1].set_ylabel("Largest-part diagonals", fontsize=9)
    axes[1].legend(fontsize=9, loc="lower left")
    axes[2].set_title("(c) Offline geometric quality", fontsize=9)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Overall PA")
    if quality is None or quality.empty:
        axes[2].text(0.5, 0.5, "Scoring in progress", ha="center", fontsize=9)
    else:
        missing = [pa.SHORT_SYSTEMS[s] for s in pa.REPRESENTATIVES if s not in set(quality.system)]
        if missing:
            axes[2].text(
                0.02,
                0.02,
                "Pending: " + ", ".join(missing),
                transform=axes[2].transAxes,
                fontsize=9,
            )
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_xlabel("Episode fraction")
        ax.grid(axis="y", alpha=0.12)
    return pa.save_figure(fig, "behavior", "behavior_main")
