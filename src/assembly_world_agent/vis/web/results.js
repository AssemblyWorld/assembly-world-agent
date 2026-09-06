/* Offline snapshot viewer. Three.js r180 + OrbitControls (MIT) are bundled locally. */
(() => {
  "use strict";
  const { THREE: T, OrbitControls } = window.ResultThree;
  const manifest = JSON.parse(document.getElementById("manifest").textContent);
  const $ = (id) => document.getElementById(id);
  $("subtitle").textContent =
    `${manifest.run} · ${manifest.samples.length} samples · Snapshot ${manifest.generated}`;
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const button = (text, action, parent) => {
    const b = el("button", "", text);
    b.onclick = action;
    parent.append(b);
    return b;
  };
  const renderer = new T.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x1b2736);
  // One WebGL context renders into visible 2D canvases, avoiding context limits.
  const rows = [];
  let mode = "all",
    follow = true,
    showColors = true;
  function partColor(index) {
    return showColors
      ? new T.Color().setHSL((0.08 + index * 0.61803398875) % 1, 0.65, 0.58)
      : new T.Color(0xa8bdcc);
  }
  const zoom = $("zoom");
  $("close-zoom").onclick = () => zoom.close();
  function fit(v) {
    const box = new T.Box3().setFromObject(v.group, true),
      center = box.getCenter(new T.Vector3());
    const direction = new T.Vector3(1.3, -1.8, 1.5).normalize();
    const right = new T.Vector3(0, 0, 1).cross(direction).normalize();
    const up = direction.clone().cross(right);
    const aspect = v.view.clientWidth / v.view.clientHeight;
    const tangent = Math.tan(T.MathUtils.degToRad(v.camera.fov / 2));
    let distance = 0.1;
    for (const x of [box.min.x, box.max.x])
      for (const y of [box.min.y, box.max.y])
        for (const z of [box.min.z, box.max.z]) {
          const point = new T.Vector3(x, y, z).sub(center);
          distance = Math.max(
            distance,
            Math.abs(point.dot(right)) / (tangent * aspect) +
              point.dot(direction),
            Math.abs(point.dot(up)) / tangent + point.dot(direction),
          );
        }
    v.camera.position
      .copy(center)
      .add(direction.multiplyScalar(distance * 1.15));
    v.controls.target.copy(center);
    v.controls.update();
  }
  function camera(v, frame) {
    v.camera.position.fromArray(frame.position);
    v.controls.target.fromArray(frame.target);
    v.controls.update();
  }
  function poses(v, values) {
    v.group.children.forEach((m, i) => {
      const p = values[i];
      m.position.fromArray(p);
      m.quaternion.set(p[4], p[5], p[6], p[3]);
    });
    v.group.updateMatrixWorld(true);
  }
  function viewer(container, geometries, row, replay) {
    const view = el("div", "view");
    container.append(view);
    const canvas = el("canvas");
    view.append(canvas);
    const scene = new T.Scene(),
      group = new T.Group();
    scene.add(group);
    const cam = new T.PerspectiveCamera(38, 1, 0.001, 10000);
    cam.up.set(0, 0, 1);
    const controls = new OrbitControls(cam, view);
    controls.enableDamping = false;
    scene.add(new T.AmbientLight(0xffffff, 2));
    const light = new T.DirectionalLight(0xffffff, 3);
    light.position.set(3, -4, 6);
    scene.add(light);
    geometries.forEach((g, index) => {
      const material = new T.MeshStandardMaterial({
        color: partColor(index),
        roughness: 0.72,
        metalness: 0.06,
        side: T.DoubleSide,
      });
      group.add(new T.Mesh(g, material));
    });
    const v = {
      view,
      canvas,
      scene,
      group,
      camera: cam,
      controls,
      context: canvas.getContext("2d"),
    };
    controls.addEventListener("start", () => {
      if (replay) {
        row.follow = false;
        row.followBox.checked = false;
      }
    });
    return v;
  }
  function timeline(r) {
    return mode === "all"
      ? [-1, ...Array.from({ length: r.meta.call_count }, (_, i) => i)]
      : [-1, ...r.meta.changed_calls];
  }
  function update(r) {
    r.steps = timeline(r);
    if (!r.steps.includes(r.current))
      r.current = r.steps.filter((i) => i <= r.current).at(-1) ?? -1;
    if (!r.data?.replay) return;
    const call = r.current < 0 ? null : r.data.replay.calls[r.current];
    const frame = r.data.replay.frames[String(call?.state_index ?? 0)];
    poses(r.replay, frame.poses);
    if (r.follow) camera(r.replay, frame.camera);
    r.slider.max = r.steps.length - 1;
    r.slider.value = r.steps.indexOf(r.current);
    r.stepLabel.textContent = `${Number(r.slider.value)} / ${r.steps.length - 1} · ${call ? `Call ${call.index + 1} · state ${call.state_index}` : "Initial state"}`;
    r.playButton.textContent = r.playing ? "Pause" : "Play";
    r.call.replaceChildren(el("strong", "", call?.name ?? "Initial state"));
    if (call) {
      r.call.append(
        el("p", "", `${call.status} · ${call.timestamp ?? ""}`),
        el("pre", "", JSON.stringify(call.arguments, null, 2)),
      );
      for (const field of ["result", "error"])
        if (call[field] !== undefined) {
          const d = el("details");
          d.append(
            el("summary", "", field === "result" ? "Return value" : "Error"),
            el("pre", "", JSON.stringify(call[field], null, 2)),
          );
          r.call.append(d);
        }
    }
    r.call.append(el("small", "", `${frame.groups.length} recorded groups`));
  }
  function move(r, step) {
    if (!r.data?.replay) return;
    r.playing = false;
    r.current =
      r.steps[
        Math.max(
          0,
          Math.min(r.steps.length - 1, r.steps.indexOf(r.current) + step),
        )
      ];
    update(r);
  }
  async function decode(index) {
    const b = Uint8Array.from(
      atob($(`sample-${index}`).textContent.trim()),
      (c) => c.charCodeAt(0),
    );
    const stream = new Blob([b])
      .stream()
      .pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(stream).text());
  }
  async function load(r) {
    if (r.loading || r.data) return;
    r.loading = true;
    try {
      r.data = await decode(r.index);
      const data = r.data;
      r.body.replaceChildren();
      for (const e of r.scroll.querySelectorAll(
        ":scope > .error, :scope > .source",
      ))
        e.remove();
      const cells = [
        "Manual",
        "Ground truth",
        "Recorded episode",
        "MCP call",
      ].map((title) => {
        const c = el("div", "cell");
        c.append(el("div", "title", title));
        r.body.append(c);
        return c;
      });
      if (data.manual.length) {
        const image = el("img", "manual-image");
        image.alt = `${data.id} manual`;
        cells[0].append(image);
        let page = r.manualPage ?? 0;
        const c = el("div", "controls");
        cells[0].append(c);
        const label = el("span");
        const show = () => {
          r.manualPage = page;
          const p = data.manual[page];
          image.removeAttribute("src");
          if (p.data) image.src = p.data;
          image.alt = p.error ?? `${data.id} · page ${page + 1}`;
          label.textContent = `${page + 1} / ${data.manual.length}`;
        };
        button(
          "Previous",
          () => {
            page = Math.max(0, page - 1);
            show();
          },
          c,
        );
        c.append(label);
        button(
          "Next",
          () => {
            page = Math.min(data.manual.length - 1, page + 1);
            show();
          },
          c,
        );
        show();
        image.onclick = () => {
          if (image.getAttribute("src")) {
            zoom.querySelector("img").src = image.src;
            zoom.showModal();
          }
        };
      } else cells[0].append(el("div", "empty", "No manual available"));
      const parts = data.replay?.parts ?? data.parts ?? [];
      const geometries = parts.map((p) => {
        const g = new T.BufferGeometry();
        g.setAttribute(
          "position",
          new T.Float32BufferAttribute(p.geometry.vertices.flat(), 3),
        );
        g.setIndex(p.geometry.triangles);
        g.computeVertexNormals();
        return g;
      });
      if (data.gt) {
        r.gt = viewer(cells[1], geometries, r, false);
        poses(r.gt, data.gt.poses);
        fit(r.gt);
        const c = el("div", "controls");
        cells[1].append(c);
        button("Reset view", () => fit(r.gt), c);
      } else cells[1].append(el("div", "empty", "Ground truth unavailable"));
      if (data.replay) {
        r.replay = viewer(cells[2], geometries, r, true);
        const c = el("div", "controls");
        cells[2].append(c);
        button("Previous", () => move(r, -1), c);
        r.playButton = button(
          "Play",
          () => {
            r.playing = !r.playing;
            r.last = performance.now();
            update(r);
          },
          c,
        );
        button("Next", () => move(r, 1), c);
        button(
          "Reset view",
          () => {
            if (r.follow)
              camera(
                r.replay,
                data.replay.frames[
                  String(
                    r.current < 0
                      ? 0
                      : data.replay.calls[r.current].state_index,
                  )
                ].camera,
              );
            else fit(r.replay);
          },
          c,
        );
        const label = el("label");
        r.followBox = el("input");
        r.followBox.type = "checkbox";
        r.followBox.checked = r.follow;
        r.followBox.onchange = () => {
          r.follow = r.followBox.checked;
          update(r);
        };
        label.append(r.followBox, document.createTextNode("Follow camera"));
        c.append(label);
        r.slider = el("input", "timeline");
        r.slider.type = "range";
        r.slider.min = 0;
        r.slider.step = 1;
        r.slider.oninput = () => {
          r.playing = false;
          r.current = r.steps[Number(r.slider.value)];
          update(r);
        };
        cells[2].append(r.slider);
        r.stepLabel = el("small");
        cells[2].append(r.stepLabel);
        r.call = el("div", "call");
        cells[3].append(r.call);
        update(r);
      } else {
        cells[2].append(el("div", "empty", "No recorded episode"));
        cells[3].append(el("div", "empty", "No recorded calls"));
      }
      if (data.errors.length)
        r.scroll.append(el("div", "error", data.errors.join("\n")));
      r.scroll.append(
        el(
          "div",
          "source",
          data.episode_source
            ? data.episode_source.path
            : "No final episode or valid checkpoint at snapshot time",
        ),
      );
      for (const [kind, v] of [
        ["gt", r.gt],
        ["replay", r.replay],
      ])
        if (v && r.savedViews?.[kind] && (kind === "gt" || !r.follow))
          camera(v, r.savedViews[kind]);
    } catch (e) {
      r.body.replaceChildren(
        el("div", "error", `Unable to load sample: ${e.message}`),
      );
    } finally {
      r.loading = false;
      if (!r.visible) unload(r);
    }
  }
  function unload(r) {
    if (r.loading || !r.data) return;
    r.savedViews = {};
    const geometries = new Set();
    for (const [kind, v] of [
      ["gt", r.gt],
      ["replay", r.replay],
    ])
      if (v) {
        r.savedViews[kind] = {
          position: v.camera.position.toArray(),
          target: v.controls.target.toArray(),
        };
        v.controls.dispose();
        v.group.children.forEach((m) => {
          geometries.add(m.geometry);
          m.material.dispose();
        });
        v.canvas.width = v.canvas.height = 1;
      }
    geometries.forEach((g) => g.dispose());
    r.data = null;
    r.gt = null;
    r.replay = null;
    r.followBox = null;
    r.body.style.minHeight = `${r.body.offsetHeight}px`;
    r.body.replaceChildren(el("div", "loading", "Scroll to load sample…"));
  }
  const observer = new IntersectionObserver(
    (entries) =>
      entries.forEach((e) => {
        const r = rows[Number(e.target.dataset.index)];
        r.visible = e.isIntersecting;
        if (r.visible) load(r);
        else unload(r);
      }),
    { rootMargin: "300px" },
  );
  manifest.samples.forEach((s, index) => {
    const section = el("section", "sample");
    section.dataset.index = index;
    const h = el("h2", "", s.id);
    h.append(el("span", "badge", s.status));
    section.append(h);
    const scroll = el("div", "scroll"),
      body = el("div", "columns");
    body.append(el("div", "loading", "Scroll to load sample…"));
    scroll.append(body);
    section.append(scroll);
    $("samples").append(section);
    const r = {
      index,
      section,
      body,
      scroll,
      meta: s,
      visible: false,
      follow,
      playing: false,
      current: s.call_count - 1,
      last: performance.now(),
    };
    r.steps = timeline(r);
    rows.push(r);
    observer.observe(section);
  });
  function endpoint(last) {
    rows.forEach((r) => {
      r.playing = false;
      r.current = last ? r.steps.at(-1) : -1;
      update(r);
    });
  }
  $("first").onclick = () => endpoint(false);
  $("last").onclick = () => endpoint(true);
  $("loop").onclick = () => {
    rows.forEach((r) => {
      r.playing = true;
      r.last = performance.now();
      if (r.data?.replay) update(r);
    });
  };
  $("pause").onclick = () => {
    rows.forEach((r) => {
      r.playing = false;
      update(r);
    });
  };
  $("mode").onchange = () => {
    mode = $("mode").value;
    rows.forEach(update);
  };
  $("follow").onchange = () => {
    follow = $("follow").checked;
    rows.forEach((r) => {
      r.follow = follow;
      if (r.followBox) r.followBox.checked = follow;
      update(r);
    });
  };
  $("colors").onchange = () => {
    showColors = $("colors").checked;
    rows.forEach((r) => {
      for (const v of [r.gt, r.replay])
        v?.group.children.forEach((mesh, index) => {
          mesh.material.color.copy(partColor(index));
        });
    });
  };
  function render(now) {
    for (const r of rows) {
      if (r.playing && r.meta.call_count && now - r.last >= 1000) {
        const count = Math.floor((now - r.last) / 1000);
        r.last += count * 1000;
        r.current =
          r.steps[(r.steps.indexOf(r.current) + count) % r.steps.length];
        update(r);
      }
      if (!r.visible) continue;
      for (const v of [r.gt, r.replay]) {
        if (!v) continue;
        const rect = v.view.getBoundingClientRect();
        if (rect.bottom < 0 || rect.top > innerHeight || !rect.width) continue;
        const w = Math.round(rect.width),
          h = Math.round(rect.height);
        renderer.setSize(w, h, false);
        v.camera.aspect = w / h;
        v.camera.updateProjectionMatrix();
        renderer.render(v.scene, v.camera);
        if (
          v.canvas.width !== renderer.domElement.width ||
          v.canvas.height !== renderer.domElement.height
        ) {
          v.canvas.width = renderer.domElement.width;
          v.canvas.height = renderer.domElement.height;
        }
        v.context.drawImage(renderer.domElement, 0, 0);
      }
    }
    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
  window.resultViewer = { rows, manifest, renderer };
})();
