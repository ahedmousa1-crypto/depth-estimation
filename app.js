/* ============================================================
   Depth estimation demo — front end
   Static. Runs on GitHub Pages. Talks to a Modal GPU endpoint.
   ============================================================ */

   const CONFIG = {
    // Paste the URL Modal prints when you run `modal deploy modal_app.py`
    ENDPOINT: 'https://YOUR-WORKSPACE--depth-demo-web.modal.run/predict',
  
    // Optional shared token. Must match SHARED_TOKEN in modal_app.py.
    // Visible to anyone who opens devtools, so it deters crawlers, nothing more.
    TOKEN: '',
  
    // Longest edge sent to the server. Smaller means faster upload and inference.
    MAX_SIDE: 1024,
  
    // How long before we tell the user the container is probably cold.
    COLD_HINT_MS: 6000,
  };
  
  /* ---------- colour ramps ---------- */
  
  const RAMPS = {
    inferno: ['#000004','#1B0C41','#4A0C6B','#781C6D','#A52C60','#CF4446','#ED6925','#FB9E3A','#FCFFA4'],
    turbo:   ['#30123B','#4145AB','#4675ED','#39A2FC','#1BCFD4','#24ECA6','#A4FC3C','#F1CA3A','#FE922A','#EA4F0D','#7A0403'],
    gray:    ['#000000','#FFFFFF'],
  };
  
  function hexToRgb(hex) {
    const n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  
  function buildLut(stops) {
    const pts = stops.map(hexToRgb);
    const lut = new Uint8ClampedArray(256 * 3);
    const seg = pts.length - 1;
    for (let i = 0; i < 256; i++) {
      const t = (i / 255) * seg;
      const a = Math.min(Math.floor(t), seg - 1);
      const f = t - a;
      for (let c = 0; c < 3; c++) {
        lut[i * 3 + c] = pts[a][c] + (pts[a + 1][c] - pts[a][c]) * f;
      }
    }
    return lut;
  }
  
  const LUTS = Object.fromEntries(Object.entries(RAMPS).map(([k, v]) => [k, buildLut(v)]));
  
  /* ---------- element handles ---------- */
  
  const el = {
    viewer:     document.getElementById('viewer'),
    dropzone:   document.getElementById('dropzone'),
    compare:    document.getElementById('compare'),
    sourceImg:  document.getElementById('sourceImg'),
    depthCanvas:document.getElementById('depthCanvas'),
    wipeHandle: document.getElementById('wipeHandle'),
    scan:       document.getElementById('scan'),
    status:     document.getElementById('status'),
    statusText: document.getElementById('statusText'),
    fileInput:  document.getElementById('fileInput'),
    cameraInput:document.getElementById('cameraInput'),
    exampleBtn: document.getElementById('exampleBtn'),
    downloadBtn:document.getElementById('downloadBtn'),
    readout:    document.getElementById('readout'),
    mInfer:     document.getElementById('mInfer'),
    mTotal:     document.getElementById('mTotal'),
    mRes:       document.getElementById('mRes'),
    mDev:       document.getElementById('mDev'),
  };
  
  const state = {
    gray: null,     // Uint8ClampedArray, one value per pixel
    w: 0,
    h: 0,
    ramp: 'inferno',
    busy: false,
    exampleIndex: 0,
    examples: null,
  };
  
  /* ---------- status line ---------- */
  
  function setStatus(text, tone = '') {
    el.statusText.textContent = text;
    el.status.dataset.tone = tone;
  }
  
  /* ---------- input handling ---------- */
  
  function wireInput(input) {
    input.addEventListener('change', () => {
      if (input.files && input.files[0]) handleFile(input.files[0]);
      input.value = '';
    });
  }
  
  wireInput(el.fileInput);
  wireInput(el.cameraInput);
  
  ['dragenter', 'dragover'].forEach(evt =>
    el.viewer.addEventListener(evt, e => {
      e.preventDefault();
      el.viewer.dataset.state = 'over';
    })
  );
  
  ['dragleave', 'drop'].forEach(evt =>
    el.viewer.addEventListener(evt, e => {
      e.preventDefault();
      if (el.viewer.dataset.state === 'over') {
        el.viewer.dataset.state = state.gray ? 'result' : 'empty';
      }
    })
  );
  
  el.viewer.addEventListener('drop', e => {
    const f = e.dataTransfer?.files?.[0];
    if (f) handleFile(f);
  });
  
  async function handleFile(file) {
    if (state.busy) return;
  
    if (!file.type.startsWith('image/')) {
      setStatus('That is not an image file. Use a JPEG, PNG or HEIC photo.', 'error');
      return;
    }
  
    try {
      const { dataUrl, width, height } = await downscale(file, CONFIG.MAX_SIDE);
      showSource(dataUrl);
      el.mRes.textContent = `${width}\u00D7${height}`;
      await predict(dataUrl);
    } catch (err) {
      console.error(err);
      setStatus('Could not read that image. Try a different file.', 'error');
    }
  }
  
  /* Resize in the browser before upload. Also strips EXIF orientation
     quirks by rasterising through a canvas. */
  function downscale(file, maxSide) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        const scale = Math.min(1, maxSide / Math.max(img.width, img.height));
        const w = Math.round(img.width * scale);
        const h = Math.round(img.height * scale);
        const c = document.createElement('canvas');
        c.width = w;
        c.height = h;
        c.getContext('2d').drawImage(img, 0, 0, w, h);
        resolve({ dataUrl: c.toDataURL('image/jpeg', 0.92), width: w, height: h });
      };
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('decode failed')); };
      img.src = url;
    });
  }
  
  function showSource(dataUrl) {
    el.sourceImg.src = dataUrl;
    el.dropzone.hidden = true;
    el.compare.hidden = false;
    el.viewer.dataset.state = 'result';
  }
  
  /* ---------- inference ---------- */
  
  async function predict(dataUrl) {
    state.busy = true;
    el.scan.hidden = false;
    el.downloadBtn.disabled = true;
    setStatus('Sending to the GPU\u2026', 'busy');
  
    const started = performance.now();
    const coldHint = setTimeout(
      () => setStatus('Still waiting. The container was asleep and is starting up\u2026', 'busy'),
      CONFIG.COLD_HINT_MS
    );
  
    try {
      const headers = { 'Content-Type': 'application/json' };
      if (CONFIG.TOKEN) headers['X-Demo-Token'] = CONFIG.TOKEN;
  
      const res = await fetch(CONFIG.ENDPOINT, {
        method: 'POST',
        headers,
        body: JSON.stringify({ image: dataUrl.split(',')[1] }),
      });
  
      if (!res.ok) {
        const detail = await res.text().catch(() => '');
        throw new Error(`Server returned ${res.status}. ${detail.slice(0, 160)}`);
      }
  
      const data = await res.json();
      await renderDepth(data);
  
      const total = Math.round(performance.now() - started);
      el.mInfer.textContent = data.inference_ms != null ? `${Math.round(data.inference_ms)} ms` : '\u2014';
      el.mTotal.textContent = `${total} ms`;
      el.mDev.textContent = data.device || '\u2014';
      el.readout.hidden = false;
      el.downloadBtn.disabled = false;
      setStatus('Depth map ready. Drag the divider to compare.', 'ok');
  
    } catch (err) {
      console.error(err);
      setStatus(explainError(err), 'error');
    } finally {
      clearTimeout(coldHint);
      el.scan.hidden = true;
      state.busy = false;
    }
  }
  
  function explainError(err) {
    if (err instanceof TypeError) {
      return 'Could not reach the server. Either it is offline, or the browser blocked the '
           + 'request because the endpoint does not allow this origin (CORS). Check the console.';
    }
    return err.message || 'Something went wrong during inference.';
  }
  
  /* ---------- rendering ---------- */
  
  async function renderDepth(data) {
    const img = await loadImage('data:image/png;base64,' + data.depth);
    const off = document.createElement('canvas');
    off.width = img.width;
    off.height = img.height;
    const octx = off.getContext('2d', { willReadFrequently: true });
    octx.drawImage(img, 0, 0);
  
    const px = octx.getImageData(0, 0, img.width, img.height).data;
    const gray = new Uint8ClampedArray(img.width * img.height);
    for (let i = 0, j = 0; i < px.length; i += 4, j++) gray[j] = px[i];
  
    state.gray = gray;
    state.w = img.width;
    state.h = img.height;
  
    paint();
  }
  
  function paint() {
    if (!state.gray) return;
    const { w, h, gray } = state;
    el.depthCanvas.width = w;
    el.depthCanvas.height = h;
  
    const ctx = el.depthCanvas.getContext('2d');
    const out = ctx.createImageData(w, h);
    const lut = LUTS[state.ramp];
  
    for (let j = 0, i = 0; j < gray.length; j++, i += 4) {
      const v = gray[j] * 3;
      out.data[i]     = lut[v];
      out.data[i + 1] = lut[v + 1];
      out.data[i + 2] = lut[v + 2];
      out.data[i + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
  }
  
  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = reject;
      im.src = src;
    });
  }
  
  /* ---------- ramp switching ---------- */
  
  document.querySelectorAll('input[name="cmap"]').forEach(r =>
    r.addEventListener('change', () => {
      state.ramp = r.value;
      paint();
    })
  );
  
  /* ---------- wipe divider ---------- */
  
  function setWipe(pct) {
    const v = Math.max(0, Math.min(100, pct));
    el.compare.style.setProperty('--wipe', v + '%');
    el.wipeHandle.setAttribute('aria-valuenow', Math.round(v));
  }
  
  setWipe(50);
  
  function wipeFromEvent(e) {
    const rect = el.compare.getBoundingClientRect();
    setWipe(((e.clientX - rect.left) / rect.width) * 100);
  }
  
  let dragging = false;
  
  el.compare.addEventListener('pointerdown', e => {
    dragging = true;
    el.compare.setPointerCapture(e.pointerId);
    wipeFromEvent(e);
  });
  
  el.compare.addEventListener('pointermove', e => {
    if (dragging) wipeFromEvent(e);
  });
  
  ['pointerup', 'pointercancel'].forEach(evt =>
    el.compare.addEventListener(evt, () => { dragging = false; })
  );
  
  el.wipeHandle.addEventListener('keydown', e => {
    const cur = Number(el.wipeHandle.getAttribute('aria-valuenow'));
    const step = e.shiftKey ? 10 : 2;
    if (e.key === 'ArrowLeft')  { setWipe(cur - step); e.preventDefault(); }
    if (e.key === 'ArrowRight') { setWipe(cur + step); e.preventDefault(); }
    if (e.key === 'Home')       { setWipe(0);   e.preventDefault(); }
    if (e.key === 'End')        { setWipe(100); e.preventDefault(); }
  });
  
  /* ---------- examples : the offline fallback ---------- */
  
  el.exampleBtn.addEventListener('click', async () => {
    if (state.busy) return;
  
    if (!state.examples) {
      try {
        const res = await fetch('examples/manifest.json');
        if (!res.ok) throw new Error();
        state.examples = await res.json();
      } catch {
        setStatus('No examples bundled yet. See the README for how to add them.', 'error');
        state.examples = [];
        return;
      }
    }
  
    if (!state.examples.length) {
      setStatus('No examples bundled yet. See the README for how to add them.', 'error');
      return;
    }
  
    const ex = state.examples[state.exampleIndex % state.examples.length];
    state.exampleIndex++;
  
    try {
      const [photo, depth] = await Promise.all([
        loadImage('examples/' + ex.photo),
        loadImage('examples/' + ex.depth),
      ]);
  
      el.sourceImg.src = photo.src;
      el.dropzone.hidden = true;
      el.compare.hidden = false;
      el.viewer.dataset.state = 'result';
  
      const off = document.createElement('canvas');
      off.width = depth.width;
      off.height = depth.height;
      const octx = off.getContext('2d', { willReadFrequently: true });
      octx.drawImage(depth, 0, 0);
      const px = octx.getImageData(0, 0, depth.width, depth.height).data;
      const gray = new Uint8ClampedArray(depth.width * depth.height);
      for (let i = 0, j = 0; i < px.length; i += 4, j++) gray[j] = px[i];
  
      state.gray = gray;
      state.w = depth.width;
      state.h = depth.height;
      paint();
  
      el.mRes.textContent = `${photo.naturalWidth}\u00D7${photo.naturalHeight}`;
      el.mInfer.textContent = 'precomputed';
      el.mTotal.textContent = '\u2014';
      el.mDev.textContent = 'bundled';
      el.readout.hidden = false;
      el.downloadBtn.disabled = false;
      setStatus(`Example: ${ex.label || ex.photo}. No GPU call was made.`, 'ok');
  
    } catch {
      setStatus('Could not load that example. Check the files in examples/.', 'error');
    }
  });
  
  /* ---------- download ---------- */
  
  el.downloadBtn.addEventListener('click', () => {
    if (!state.gray) return;
    const a = document.createElement('a');
    a.download = `depth-${state.ramp}-${Date.now()}.png`;
    a.href = el.depthCanvas.toDataURL('image/png');
    a.click();
  });
  
  /* ---------- first-run sanity check ---------- */
  
  if (CONFIG.ENDPOINT.includes('YOUR-WORKSPACE')) {
    setStatus('No endpoint configured yet. Set CONFIG.ENDPOINT in app.js.', 'error');
  }