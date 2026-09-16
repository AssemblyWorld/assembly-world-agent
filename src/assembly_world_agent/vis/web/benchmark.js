/* AssemblyWorldBench preview. Three.js r180 + OrbitControls (MIT) are bundled locally. */
(() => {
  "use strict";
  const { THREE: T, OrbitControls } = window.ResultThree;
  const page = JSON.parse(document.getElementById("manifest").textContent);
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  $("subtitle").textContent =
    `${page.statistics.evaluations} evaluations · ${page.statistics.distinct_shapes} distinct shapes · ` +
    `generated ${page.generated}`;

  // --- decoding -------------------------------------------------------------------
  async function decode(scriptId) {
    const b64 = document.getElementById(scriptId).textContent.trim();
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(stream).text());
  }
  const cache = new Map();
  async function geometryOf(shape) {
    if (!cache.has(shape.script)) cache.set(shape.script, decode(shape.script));
    return cache.get(shape.script);
  }

  // --- filters and list -----------------------------------------------------------
  const shapes = page.shapes;
  const options = (id, values, all) => {
    const select = $(id);
    select.append(new Option(all, ""));
    for (const v of values) select.append(new Option(v, v));
    select.onchange = renderList;
  };
  const label = (value) => (value == null ? "(none)" : String(value));
  const unique = (key) => [...new Set(shapes.flatMap((s) => (Array.isArray(s[key]) ? s[key] : [s[key]])).map(label))].sort();
  options("f-block", unique("blocks"), "all blocks");
  options("f-category", unique("category"), "all categories");
  options("f-band", unique("band"), "all bands");
  let visible = [];
  let current = null;
  function renderList() {
    const block = $("f-block").value, category = $("f-category").value, band = $("f-band").value;
    visible = shapes.filter(
      (s) =>
        (!block || s.blocks.includes(block)) &&
        (!category || label(s.category) === category) &&
        (!band || s.band === band),
    );
    $("f-count").textContent = `${visible.length} / ${shapes.length}`;
    const list = $("list");
    list.replaceChildren();
    for (const shape of visible) {
      const item = el("div", "item");
      item.dataset.script = shape.script;
      item.append(el("span", "id", shape.sample_id));
      item.append(el("span", `badge ${shape.band}`, shape.band));
      item.append(
        el("span", "meta", `${shape.source}${shape.category ? " · " + shape.category : ""} · ${shape.parts} parts` +
          (shape.decimated ? " · decimated" : "")),
      );
      item.onclick = () => show(shape);
      if (current && current.script === shape.script) item.classList.add("selected");
      list.append(item);
    }
    if (visible.length && !(current && visible.includes(current))) show(visible[0]);
  }

  // --- three.js ------------------------------------------------------------------
  const canvas = $("canvas");
  const renderer = new T.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x1b2736);
  const scene = new T.Scene();
  const camera = new T.PerspectiveCamera(45, 1, 0.001, 1000);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  scene.add(new T.HemisphereLight(0xdfe9f5, 0x2a3442, 1.1));
  const sun = new T.DirectionalLight(0xffffff, 1.4);
  sun.position.set(2, -3, 4);
  scene.add(sun);
  const grid = new T.GridHelper(4, 40, 0x3b4a5e, 0x263243);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);
  let group = null;
  let blend = 0;
  let showColors = true;
  const partColor = (i, n) =>
    showColors ? new T.Color().setHSL((0.08 + i * 0.61803398875) % 1, 0.65, 0.58) : new T.Color(0xa8bdcc);

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe($("stage"));

  function poseOf(part, t) {
    const a = part.userData.initial, b = part.userData.gt;
    const pa = new T.Vector3(a[0], a[1], a[2]), pb = new T.Vector3(b[0], b[1], b[2]);
    const qa = new T.Quaternion(a[4], a[5], a[6], a[3]), qb = new T.Quaternion(b[4], b[5], b[6], b[3]);
    part.position.copy(pa.lerp(pb, t));
    part.quaternion.copy(qa.slerp(qb, t));
  }
  function applyBlend(t) {
    blend = t;
    $("blend").value = Math.round(t * 1000);
    if (group) for (const part of group.children) poseOf(part, t);
    $("b-initial").classList.toggle("active", t === 0);
    $("b-gt").classList.toggle("active", t === 1);
  }

  function buildGroup(data) {
    const g = new T.Group();
    let extent = 0;
    data.parts.forEach((p, i) => {
      const positions = new Float32Array(p.vertices.flat());
      const geometry = new T.BufferGeometry();
      geometry.setAttribute("position", new T.BufferAttribute(positions, 3));
      geometry.computeBoundingSphere();
      extent = Math.max(extent, geometry.boundingSphere.radius);
      geometry.setIndex(new T.BufferAttribute(new Uint32Array(p.triangles), 1));
      geometry.computeVertexNormals();
      const object = new T.Mesh(
        geometry,
        new T.MeshStandardMaterial({ color: partColor(i, data.parts.length), flatShading: true, side: T.DoubleSide, roughness: 0.7 }),
      );
      object.userData = { initial: data.initial[i], gt: data.gt[i], index: i, id: p.id, decimated: p.decimated };
      g.add(object);
    });
    return g;
  }

  function fitBoth() {
    if (!group) return;
    const box = new T.Box3();
    for (const t of [0, 1]) {
      for (const part of group.children) poseOf(part, t);
      group.updateMatrixWorld(true);
      box.union(new T.Box3().setFromObject(group, true));
    }
    applyBlend(blend);
    const center = box.getCenter(new T.Vector3());
    const size = box.getSize(new T.Vector3()).length() || 1;
    const direction = new T.Vector3(1.3, -1.8, 1.2).normalize();
    camera.position.copy(center).add(direction.multiplyScalar(size * 0.9));
    camera.near = size / 1000;
    camera.far = size * 50;
    camera.updateProjectionMatrix();
    controls.target.copy(center);
    controls.update();
    grid.position.set(center.x, center.y, 0);
    const cells = 40, span = Math.max(size * 1.5, 1);
    grid.scale.setScalar(span / 4);
    grid.visible = $("grid").checked;
  }

  function recolor() {
    if (!group) return;
    group.children.forEach((o, i) => o.material.color.copy(partColor(i, group.children.length)));
    renderLegend();
  }
  function renderLegend() {
    const legend = $("legend");
    legend.replaceChildren();
    if (!group) return;
    group.children.forEach((o, i) => {
      const item = el("span");
      const swatch = el("i");
      swatch.style.background = "#" + partColor(i, group.children.length).getHexString();
      item.append(swatch, document.createTextNode(`${o.userData.id}${o.userData.decimated ? " (decimated)" : ""}`));
      legend.append(item);
    });
  }

  async function show(shape) {
    current = shape;
    for (const item of $("list").children) item.classList.toggle("selected", item.dataset.script === shape.script);
    const data = await geometryOf(shape);
    if (current !== shape) return;
    if (group) {
      scene.remove(group);
      for (const o of group.children) { o.geometry.dispose(); o.material.dispose(); }
    }
    group = buildGroup(data);
    scene.add(group);
    fitBoth();
    renderDetails(shape);
    renderLegend();
  }

  function renderDetails(shape) {
    const dl = $("details");
    dl.replaceChildren();
    const rows = [
      ["Sample", shape.sample_id],
      ["Source", shape.source],
      ["Dataset", `${shape.repo_id} @ ${shape.revision.slice(0, 12)}`],
      ["Category", shape.category ?? "—"],
      ["Parts", String(shape.parts)],
      ["Band", shape.band],
      ["Blocks", shape.blocks.map((b, i) => `${b} [${shape.reference_modes[i]}]`).join(", ")],
      ["Data root", shape.data],
      ["Geometry", `${shape.shown_triangles.toLocaleString()} triangles shown · ${shape.source_triangles.toLocaleString()} in source` +
        (shape.decimated ? " · display-only decimation" : "")],
      ["Initial sha256", shape.sha256.slice(0, 16) + "…"],
    ];
    for (const [k, v] of rows) { dl.append(el("dt", "", k)); dl.append(el("dd", "", v)); }
  }

  // --- statistics panels -------------------------------------------------------------
  function table(headers, rows) {
    const t = el("table");
    const tr = el("tr");
    for (const h of headers) tr.append(el("th", "", h));
    t.append(tr);
    for (const r of rows) {
      const row = el("tr");
      r.forEach((c, i) => row.append(el("td", i ? "num" : "", String(c))));
      t.append(row);
    }
    return t;
  }
  const stats = page.statistics;
  const statRows = Object.entries(stats.sources).map(([name, s]) => [
    name,
    s.shapes,
    Object.entries(s.band_counts).map(([k, v]) => `${k} ${v}`).join(" / "),
    Object.entries(s.categories).map(([k, v]) => `${k} ${v}`).join(", "),
  ]);
  $("stats").append(table(["source", "shapes", "bands", "categories"], statRows));
  const pooled = {};
  for (const s of Object.values(stats.sources)) for (const [k, v] of Object.entries(s.band_counts)) pooled[k] = (pooled[k] || 0) + v;
  $("pooled").append(table(["band", "shapes"], Object.entries(pooled)));
  $("blocks").append(
    table(["block", "source", "mode", "samples"], stats.blocks.map((b) => [b.name, b.source, b.reference_mode, b.samples])),
  );

  // --- controls ------------------------------------------------------------------------
  let playing = null;
  function stop() { if (playing) { cancelAnimationFrame(playing); playing = null; $("b-play").classList.remove("active"); } }
  $("b-initial").onclick = () => { stop(); applyBlend(0); };
  $("b-gt").onclick = () => { stop(); applyBlend(1); };
  $("blend").oninput = (e) => { stop(); applyBlend(Number(e.target.value) / 1000); };
  $("b-play").onclick = () => {
    if (playing) return stop();
    $("b-play").classList.add("active");
    const start = performance.now(), from = blend >= 1 ? 0 : blend, duration = 1400 * (1 - from);
    const step = (now) => {
      const t = Math.min(1, from + (now - start) / 1400);
      applyBlend(t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
      if (t < 1) playing = requestAnimationFrame(step); else stop();
    };
    playing = requestAnimationFrame(step);
  };
  $("colors").onchange = (e) => { showColors = e.target.checked; recolor(); };
  $("grid").onchange = (e) => { grid.visible = e.target.checked; };
  $("b-reset").onclick = fitBoth;
  addEventListener("keydown", (e) => {
    if (!visible.length || !current) return;
    const i = visible.indexOf(current);
    if (e.key === "ArrowRight" && i < visible.length - 1) show(visible[i + 1]);
    if (e.key === "ArrowLeft" && i > 0) show(visible[i - 1]);
  });

  (function loop() {
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  })();
  resize();
  renderList();
})();
