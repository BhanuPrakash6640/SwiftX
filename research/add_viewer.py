"""Insert the SiftX memory viewer into frontend/index.html.

    python add_viewer.py <index.html> <trace.json>
Idempotent: re-running replaces the embedded trace data only.
"""
import re
import sys

page, trace_path = sys.argv[1], sys.argv[2]
s = open(page).read()
trace = open(trace_path).read().replace("</", "<\\/")

if 'id="trace-data"' in s:
    s = re.sub(r'(<script type="application/json" id="trace-data">).*?(</script>)',
               lambda m: m.group(1) + trace + m.group(2), s, count=1, flags=re.S)
    open(page, "w").write(s)
    print("trace replaced")
    sys.exit(0)

s = s.replace("  --good: #0f8a4a;\n", "  --good: #0f8a4a;\n  --k-key: #7a5bd1; --k-val: #1a9a8a; --k-q: #c08600; --k-noise: #a7b2bc;\n", 1)
s = s.replace("--good: #3cc47c;\n", "--good: #3cc47c;\n  --k-key: #9478e6; --k-val: #24a090; --k-q: #b98c12; --k-noise: #3a5269;\n", 2)
assert s.count("--k-key") == 3, "colour tokens not inserted in all three theme blocks"
s = s.replace(".controls { display: flex;", '.controls input[type="range"] { flex: 1 1 200px; min-width: 0; accent-color: var(--sift); }\n.controls { display: flex;', 1)

section = '''  <!-- 4. Inside SiftX -->
  <section aria-labelledby="s5">
    <div class="prose" style="display:grid;gap:12px">
      <span class="label">Part 4 · SiftX, our research version</span>
      <h2 id="s5">What SiftX chooses to remember</h2>
      <p>Sift's notebook is blurry: it blends everything together. SiftX adds a small <strong>shelf of exact notes</strong>. Only 32 tokens fit on the shelf, and the model decides which ones to keep. A <strong>gate</strong> then decides, word by word, whether it's worth checking the shelf at all. Skipping the check saves work.</p>
      <p>Below is one real test sequence: 16 facts (a <em>key</em> followed by its <em>value</em>), then noise with 16 questions mixed in. To answer a question, the model has to find the value that went with that key. Each row is one chunk of 16 tokens. Move the slider to watch the shelf change.</p>
    </div>
    <div class="panel">
      <div class="stage" id="stage5" role="img" aria-label="3D view: 128 tokens laid out in 8 rows of 16. Tokens copied onto a shelf above show what SiftX's memory holds while it reads the selected row; rings mark tokens where the read gate opened."><span class="hint">drag to rotate</span></div>
      <div class="readout" aria-live="polite">
        <div><span class="label">Reading row</span><span class="big" id="x-chunk">—</span><span class="note">of 8 (16 tokens each)</span></div>
        <div><span class="label">On the shelf</span><span class="big" id="x-shelf">—</span><span class="note" id="x-shelf-t">—</span></div>
        <div><span class="label">Gate opened</span><span class="big" id="x-gate">—</span><span class="note">of the tokens in this row</span></div>
        <div><span class="label">Questions in this row</span><span class="big" id="x-acc">—</span><span class="note" id="x-acc-t">answered correctly</span></div>
      </div>
      <form class="controls" id="f5" onsubmit="return false">
        <label for="x-slider" class="label">Row</label>
        <input id="x-slider" type="range" min="0" max="7" value="3">
        <button type="button" class="ghost" id="x-play">Play</button>
      </form>
    </div>
    <div class="chips">
      <span class="chip" style="--c:var(--k-key)">key</span>
      <span class="chip" style="--c:var(--k-val)">value (the answer to remember)</span>
      <span class="chip" style="--c:var(--k-q)">question</span>
      <span class="chip" style="--c:var(--k-noise)">noise</span>
      <span class="chip" style="--c:var(--ink)">ring = gate opened, the model checked its shelf</span>
    </div>
    <p class="note" id="x-src"></p>
  </section>

  <!-- 5. Results -->'''
assert "  <!-- 4. Results -->" in s
s = s.replace("  <!-- 4. Results -->", section, 1)
s = s.replace("Part 4 · What we measured", "Part 5 · What we measured", 1)

script = open(__file__.replace("add_viewer.py", "viewer.js")).read()
s = s.rstrip() + '\n<script type="application/json" id="trace-data">' + trace + '</script>\n<script>\n' + script + '\n</script>\n'
open(page, "w").write(s)
print("viewer added", len(s))
