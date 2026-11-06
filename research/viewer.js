// ---------- Part 4: SiftX memory viewer ----------
(function () {
  const el = document.getElementById("stage5");
  if (!el || !window.THREE) return;
  const data = JSON.parse(document.getElementById("trace-data").textContent);
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const C = data.chunk, NC = Math.ceil(data.seq_len / C), M = data.mem_slots;
  const kindVar = { key: "--k-key", value: "--k-val", query: "--k-q", noise: "--k-noise" };

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  el.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 200);
  const base = [0, 7.2, 11.5];
  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const dl = new THREE.DirectionalLight(0xffffff, 0.6); dl.position.set(4, 10, 6); scene.add(dl);
  const pivot = new THREE.Group(); scene.add(pivot);
  let yaw = 0, pitch = 0, drag = false, lx = 0, ly = 0;
  el.addEventListener("pointerdown", e => { drag = true; lx = e.clientX; ly = e.clientY; el.setPointerCapture(e.pointerId); });
  el.addEventListener("pointermove", e => { if (!drag) return; yaw += (e.clientX - lx) * 0.006; pitch = Math.max(-0.35, Math.min(0.35, pitch + (e.clientY - ly) * 0.004)); lx = e.clientX; ly = e.clientY; });
  el.addEventListener("pointerup", () => drag = false);
  el.addEventListener("pointercancel", () => drag = false);
  function resize() {
    const w = el.clientWidth, h = el.clientHeight;
    renderer.setSize(w, h, false); camera.aspect = w / h;
    const k = w < 600 ? 1.12 : 1;
    camera.position.set(base[0], base[1] * k, base[2] * k); camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(el); resize();

  // tokens: one row per chunk
  const SP = 0.62, RZ = 0.78;
  const tokPos = i => new THREE.Vector3(((i % C) - (C - 1) / 2) * SP, 0, (Math.floor(i / C) - (NC - 1) / 2) * RZ + 0.9);
  const geo = new THREE.BoxGeometry(0.46, 0.22, 0.5);
  const toks = data.tokens.map(t => {
    const m = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ transparent: true }));
    m.position.copy(tokPos(t.i)); pivot.add(m); return m;
  });
  const ringGeo = new THREE.TorusGeometry(0.2, 0.035, 8, 24);
  const rings = data.tokens.map(t => {
    const r = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ transparent: true }));
    r.rotation.x = Math.PI / 2; r.position.copy(tokPos(t.i)).add(new THREE.Vector3(0, 0.2, 0)); pivot.add(r); return r;
  });
  // the shelf: M slots in two rows, behind and above the tokens
  const per = Math.ceil(M / 2);
  const slotPos = j => new THREE.Vector3(((j % per) - (per - 1) / 2) * 0.34, 2.6 - Math.floor(j / per) * 0.42, -3.7);
  const shelf = new THREE.Mesh(new THREE.BoxGeometry(per * 0.34 + 0.4, 0.06, 0.6), new THREE.MeshLambertMaterial());
  shelf.position.set(0, 2.33 - 0.42, -3.7); pivot.add(shelf);
  const slotGeo = new THREE.BoxGeometry(0.28, 0.3, 0.3);
  const slots = Array.from({ length: M }, (_, j) => {
    const m = new THREE.Mesh(slotGeo, new THREE.MeshLambertMaterial({ transparent: true }));
    m.position.copy(slotPos(j)); pivot.add(m); return m;
  });
  const lineMat = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.3 });
  let lines = null;
  const band = new THREE.Mesh(new THREE.BoxGeometry(C * SP + 0.3, 0.04, RZ * 0.95), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.18 }));
  pivot.add(band);

  let chunk = Math.min(3, NC - 1), playing = false, timer = 0;
  const slider = document.getElementById("x-slider");
  slider.max = NC - 1;
  const kindCol = k => new THREE.Color(css(kindVar[k]) || "#888");
  const memory = c => (data.memory_by_chunk[c] || []).filter(i => i < c * C);   // only earlier rows are eligible

  function render() {
    const mem = memory(chunk), inMem = new Set(mem);
    const ink = new THREE.Color(css("--ink") || "#102433");
    const rule = new THREE.Color(css("--rule") || "#c9d5df");
    data.tokens.forEach((t, i) => {
      const row = Math.floor(i / C);
      toks[i].material.color.copy(kindCol(t.kind));
      toks[i].material.opacity = row === chunk ? 1 : row < chunk ? (inMem.has(i) ? 0.95 : 0.35) : 0.12;
      toks[i].userData.lift = inMem.has(i) ? 0.14 : 0;
      rings[i].material.color.copy(ink);
      rings[i].material.opacity = row === chunk && t.gate ? 0.95 : 0;
    });
    slots.forEach((sl, j) => {
      if (j < mem.length) { sl.material.color.copy(kindCol(data.tokens[mem[j]].kind)); sl.material.opacity = 1; }
      else { sl.material.color.copy(rule); sl.material.opacity = 0.35; }
    });
    shelf.material.color.copy(rule);
    band.material.color.copy(new THREE.Color(css("--sift") || "#2a78d6"));
    band.position.copy(tokPos(chunk * C)).setX(0).add(new THREE.Vector3(0, -0.14, 0));
    if (lines) { pivot.remove(lines); lines.geometry.dispose(); }
    const pts = [];
    mem.forEach((i, j) => { pts.push(tokPos(i).add(new THREE.Vector3(0, 0.14, 0)), slotPos(j)); });
    lineMat.color.copy(ink);
    lines = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), lineMat);
    pivot.add(lines);

    const cnt = { key: 0, value: 0, query: 0, noise: 0 };
    mem.forEach(i => cnt[data.tokens[i].kind]++);
    const rowT = data.tokens.slice(chunk * C, (chunk + 1) * C);
    const q = rowT.filter(t => t.kind === "query");
    document.getElementById("x-chunk").textContent = chunk + 1;
    document.getElementById("x-shelf").textContent = mem.length + " / " + M;
    document.getElementById("x-shelf-t").textContent = mem.length
      ? `${cnt.value} values, ${cnt.key} keys, ${cnt.noise} noise, ${cnt.query} questions`
      : "empty: nothing has been read before this row";
    document.getElementById("x-gate").textContent = Math.round(100 * rowT.filter(t => t.gate).length / rowT.length) + "%";
    document.getElementById("x-acc").textContent = q.length ? `${q.filter(t => t.correct).length} / ${q.length}` : "—";
    document.getElementById("x-acc-t").textContent = q.length ? "answered correctly" : "no questions in this row";
    slider.value = chunk;
  }
  slider.addEventListener("input", () => { chunk = +slider.value; render(); });
  const play = document.getElementById("x-play");
  play.addEventListener("click", () => { playing = !playing; play.textContent = playing ? "Pause" : "Play"; });
  const s = data.summary;
  document.getElementById("x-src").textContent =
    `Data: model "${data.run}" (${data.policy} memory, ${data.read_gate} gate). Over the whole sequence its gate opened on ${Math.round(100 * s.retrieval_rate)}% of tokens and on ${Math.round(100 * s.gate_on_queries)}% of questions, and it answered ${Math.round(100 * s.query_accuracy)}% of questions correctly. ` +
    (s.query_accuracy < 0.5 ? "This is a partly trained CPU pilot model. Its shelf already favours values, but it hasn't learned to use them yet. The fully trained GPU model will replace it here." : "");
  render();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
  new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    if (playing) { timer += dt; if (timer > 1.6) { timer = 0; chunk = (chunk + 1) % NC; render(); } }
    toks.forEach(m => { const y = m.userData.lift || 0; m.position.y += (y - m.position.y) * Math.min(1, dt * 6); });
    if (!reduce) rings.forEach(r => { r.rotation.z += dt * 1.5; });
    pivot.rotation.y += (yaw - pivot.rotation.y) * 0.08; pivot.rotation.x += (pitch - pivot.rotation.x) * 0.08;
    camera.lookAt(0, 0.6, -0.8); renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
