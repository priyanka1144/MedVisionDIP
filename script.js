/**
 * ════════════════════════════════════════════════════════════════
 *  MedVision DIP — Medical Image Enhancement System
 *  script.js — Complete DIP pipeline implementation
 *  All filters implemented manually using Canvas API pixel ops
 * ════════════════════════════════════════════════════════════════
 */

'use strict';

/* ─────────────────────────────────────────────
   State
───────────────────────────────────────────── */
const state = {
  originalImageData: null,   // Raw pixel data of uploaded image
  canvasW: 0,
  canvasH: 0,
  results: {},               // Keyed stage results: { laplacian, sharpened, sobelX, sobelY, sobel, smoothed, mask, final, gamma }
  gammaValue: 1.0,
  processing: false,
};

/* ─────────────────────────────────────────────
   DOM References
───────────────────────────────────────────── */
const dom = {
  fileInput:       document.getElementById('fileInput'),
  uploadZone:      document.getElementById('uploadZone'),
  controlPanel:    document.getElementById('controlPanel'),
  pipelineSection: document.getElementById('pipelineSection'),
  compareSection:  document.getElementById('compareSection'),

  originalCanvas:  document.getElementById('originalCanvas'),
  histogramCanvas: document.getElementById('histogramCanvas'),
  imageInfo:       document.getElementById('imageInfo'),

  gammaSlider:     document.getElementById('gammaSlider'),
  gammaValue:      document.getElementById('gammaValue'),
  processBtn:      document.getElementById('processBtn'),
  resetBtn:        document.getElementById('resetBtn'),
  downloadBtn:     document.getElementById('downloadBtn'),

  stagesGrid:      document.getElementById('stagesGrid'),

  compareOriginal: document.getElementById('compareOriginal'),
  compareFinal:    document.getElementById('compareFinal'),
  compareSlider:   document.getElementById('compareSlider'),
  compareContainer:document.getElementById('compareContainer'),

  progressWrap:    document.getElementById('progressWrap'),
  progressFill:    document.getElementById('progressFill'),
  progressText:    document.getElementById('progressText'),
  progressPct:     document.getElementById('progressPct'),
  progressSteps:   document.getElementById('progressSteps'),

  loadingOverlay:  document.getElementById('loadingOverlay'),
  loadingText:     document.getElementById('loadingText'),
};

/* ─────────────────────────────────────────────
   Pipeline Stage Definitions
───────────────────────────────────────────── */
const STAGES = [
  { key: 'original',  title: 'Original Image',     tag: 'INPUT',    desc: 'Raw uploaded medical image before any processing.' },
  { key: 'laplacian', title: 'Laplacian Filter',   tag: 'KERNEL',   desc: 'Second-order derivative highlighting rapid intensity changes.' },
  { key: 'sharpened', title: 'Sharpened Image',    tag: 'STAGE 1',  desc: 'Original + scaled Laplacian response — enhanced structural detail.' },
  { key: 'sobelX',    title: 'Sobel Gx (Horiz.)',  tag: 'STAGE 2A', desc: 'Horizontal gradient detecting vertical edges and boundaries.' },
  { key: 'sobelY',    title: 'Sobel Gy (Vert.)',   tag: 'STAGE 2B', desc: 'Vertical gradient detecting horizontal anatomical features.' },
  { key: 'sobel',     title: 'Sobel Magnitude',    tag: 'STAGE 2',  desc: '√(Gx²+Gy²) — combined edge magnitude map of the image.' },
  { key: 'smoothed',  title: 'Averaged (5×5)',      tag: 'STAGE 3',  desc: '5×5 uniform averaging filter for noise suppression.' },
  { key: 'mask',      title: 'Enhancement Mask',   tag: 'STAGE 4',  desc: 'Mask = Sobel × Smoothed — structural enhancement layer.' },
  { key: 'final',     title: 'Final Enhanced',     tag: 'STAGE 5',  desc: 'Final = Original + Mask — composite enhanced output.' },
  { key: 'gamma',     title: 'Gamma Corrected',    tag: 'OUTPUT',   desc: `s = c·rᵞ — power-law intensity mapping (γ = ${state.gammaValue.toFixed(2)}).` },
];

/* ════════════════════════════════════════════
   UTILITY FUNCTIONS
════════════════════════════════════════════ */

/**
 * Convert ImageData to a simple Float32Array for pixel math (0–255 range).
 * Works on grayscale: averages R,G,B channels.
 */
function imageDataToGray(imageData) {
  const { data, width, height } = imageData;
  const gray = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  return gray;
}

/**
 * Convert Float32Array (0–255) back to ImageData (grayscale).
 * Clamps values automatically.
 */
function grayToImageData(gray, width, height) {
  const imgData = new ImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const v = Math.max(0, Math.min(255, Math.round(gray[i])));
    imgData.data[i * 4]     = v;
    imgData.data[i * 4 + 1] = v;
    imgData.data[i * 4 + 2] = v;
    imgData.data[i * 4 + 3] = 255;
  }
  return imgData;
}

/**
 * Apply a convolution kernel to a grayscale Float32Array.
 * Returns a new Float32Array (can hold negative values — caller normalises).
 * @param {Float32Array} gray     - Input pixel array
 * @param {number}       width    - Image width
 * @param {number}       height   - Image height
 * @param {number[][]}   kernel   - 2D kernel (square, odd size)
 * @param {boolean}      clampResult - If true clamps 0-255, else raw output
 */
function convolve(gray, width, height, kernel, clampResult = false) {
  const kSize  = kernel.length;
  const kHalf  = Math.floor(kSize / 2);
  const output = new Float32Array(width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let ky = 0; ky < kSize; ky++) {
        for (let kx = 0; kx < kSize; kx++) {
          const py = Math.min(height - 1, Math.max(0, y + ky - kHalf));
          const px = Math.min(width  - 1, Math.max(0, x + kx - kHalf));
          acc += kernel[ky][kx] * gray[py * width + px];
        }
      }
      output[y * width + x] = clampResult ? Math.max(0, Math.min(255, acc)) : acc;
    }
  }
  return output;
}

/**
 * Normalise a Float32Array to 0-255 range.
 */
function normalise(arr) {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < min) min = arr[i];
    if (arr[i] > max) max = arr[i];
  }
  const range = max - min || 1;
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    out[i] = ((arr[i] - min) / range) * 255;
  }
  return out;
}

/**
 * Draw a Float32Array as an image on a canvas element.
 * Auto-normalises if requested.
 */
function drawGrayToCanvas(canvas, gray, width, height, doNormalise = false) {
  canvas.width  = width;
  canvas.height = height;
  const ctx  = canvas.getContext('2d');
  const data = doNormalise ? normalise(gray) : gray;
  ctx.putImageData(grayToImageData(data, width, height), 0, 0);
}

/** Sleep (for animation frames) */
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

/* ════════════════════════════════════════════
   PROGRESS HELPERS
════════════════════════════════════════════ */
const STEP_NAMES = ['Load','Laplacian','Sharpen','Sobel-X','Sobel-Y','Magnitude','Average','Mask','Composite','Gamma'];

function initProgress() {
  dom.progressWrap.style.display = 'block';
  dom.progressSteps.innerHTML = STEP_NAMES.map(n =>
    `<span class="step-chip" data-step="${n}">${n}</span>`
  ).join('');
  setProgress(0, 'Initialising…');
}

function setProgress(pct, label, activeStep) {
  dom.progressFill.style.width = pct + '%';
  dom.progressPct.textContent  = Math.round(pct) + '%';
  dom.progressText.textContent = label;
  if (activeStep) {
    dom.progressSteps.querySelectorAll('.step-chip').forEach(el => {
      if (el.dataset.step === activeStep) el.classList.add('active');
      else if (el.classList.contains('active')) {
        el.classList.remove('active');
        el.classList.add('done');
      }
    });
  }
}

/* ════════════════════════════════════════════
   DIP PIPELINE STAGES
════════════════════════════════════════════ */

/** Stage 1 — Laplacian filter */
function applyLaplacian(gray, w, h) {
  const kernel = [
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0],
  ];
  return convolve(gray, w, h, kernel, false); // raw (may be negative)
}

/** Stage 2 — Sharpened = Original + Laplacian (scaled) */
function applySharpen(gray, laplacian) {
  const out = new Float32Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    out[i] = Math.max(0, Math.min(255, gray[i] + laplacian[i]));
  }
  return out;
}

/** Stage 3a — Sobel Gx */
function applySobelX(gray, w, h) {
  const Gx = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
  ];
  return convolve(gray, w, h, Gx, false);
}

/** Stage 3b — Sobel Gy */
function applySobelY(gray, w, h) {
  const Gy = [
    [ 1,  2,  1],
    [ 0,  0,  0],
    [-1, -2, -1],
  ];
  return convolve(gray, w, h, Gy, false);
}

/** Stage 3c — Sobel magnitude = sqrt(Gx^2 + Gy^2) */
function sobelMagnitude(gx, gy) {
  const out = new Float32Array(gx.length);
  for (let i = 0; i < gx.length; i++) {
    out[i] = Math.sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
  }
  return out;
}

/** Stage 4 — 5×5 Averaging filter */
function applyAverage5x5(gray, w, h) {
  const kernel = Array.from({ length: 5 }, () => Array(5).fill(1 / 25));
  return convolve(gray, w, h, kernel, true);
}

/** Stage 5 — Mask = Sobel (normalised) × Smoothed (normalised) */
function createMask(sobelMag, smoothed) {
  // Normalise both to 0–1, multiply, scale back
  const n1 = normalise(sobelMag);
  const n2 = normalise(smoothed);
  const out = new Float32Array(sobelMag.length);
  for (let i = 0; i < sobelMag.length; i++) {
    out[i] = (n1[i] / 255) * n2[i]; // range 0–255 * weight
  }
  return out;
}

/** Stage 6 — Final = Original + Mask (scaled) */
function applyMaskToOriginal(gray, mask) {
  const normMask = normalise(mask);
  const out = new Float32Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    out[i] = Math.max(0, Math.min(255, gray[i] + normMask[i] * 0.5));
  }
  return out;
}

/** Stage 7 — Gamma correction: s = 255 * (r/255)^gamma */
function applyGamma(gray, gamma) {
  const out = new Float32Array(gray.length);
  const c   = 255;
  for (let i = 0; i < gray.length; i++) {
    out[i] = c * Math.pow(gray[i] / 255, gamma);
  }
  return out;
}

/* ════════════════════════════════════════════
   HISTOGRAM
════════════════════════════════════════════ */
function drawHistogram(gray, canvas) {
  const ctx  = canvas.getContext('2d');
  const W    = canvas.width;
  const H    = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Build frequency bins
  const bins = new Array(256).fill(0);
  for (let i = 0; i < gray.length; i++) {
    bins[Math.round(Math.min(255, Math.max(0, gray[i])))]++;
  }
  const maxBin = Math.max(...bins);

  // Background
  ctx.fillStyle = 'rgba(0,0,0,0.4)';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = 'rgba(0,210,255,0.07)';
  ctx.lineWidth   = 1;
  for (let i = 1; i < 4; i++) {
    ctx.beginPath();
    ctx.moveTo(0, H * i / 4);
    ctx.lineTo(W, H * i / 4);
    ctx.stroke();
  }

  // Bars with gradient fill
  const barW = W / 256;
  for (let b = 0; b < 256; b++) {
    const barH  = (bins[b] / maxBin) * (H - 2);
    const ratio = b / 255;
    // Gradient from dark-blue to cyan
    const r = Math.round(ratio * 0);
    const g = Math.round(ratio * 200 + 30);
    const blueC = 255;
    ctx.fillStyle = `rgba(${r},${g},${blueC},0.65)`;
    ctx.fillRect(b * barW, H - barH, barW, barH);
  }

  // Top glow line
  const grad = ctx.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0, 'rgba(0,50,150,0)');
  grad.addColorStop(0.5, 'rgba(0,210,255,0.6)');
  grad.addColorStop(1, 'rgba(0,50,150,0)');
  ctx.strokeStyle = grad;
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  for (let b = 0; b < 256; b++) {
    const barH = (bins[b] / maxBin) * (H - 2);
    const bx   = b * barW + barW / 2;
    if (b === 0) ctx.moveTo(bx, H - barH);
    else ctx.lineTo(bx, H - barH);
  }
  ctx.stroke();
}

/* ════════════════════════════════════════════
   PIPELINE RUNNER
════════════════════════════════════════════ */
async function runPipeline() {
  if (!state.originalImageData) return;
  if (state.processing) return;
  state.processing = true;
  dom.processBtn.disabled = true;
  dom.downloadBtn.disabled = true;

  // Show loading overlay briefly
  dom.loadingText.textContent = 'Starting DIP pipeline…';
  dom.loadingOverlay.style.display = 'flex';

  initProgress();

  const { width: W, height: H } = state.originalImageData;
  await sleep(400);

  // ── Stage 0: Load grayscale
  setProgress(5, 'Loading image pixels…', 'Load');
  const gray = imageDataToGray(state.originalImageData);
  state.results.original = gray;
  dom.loadingText.textContent = 'Applying Laplacian filter…';
  await sleep(200);

  // ── Stage 1: Laplacian
  setProgress(15, 'Computing Laplacian…', 'Laplacian');
  dom.loadingOverlay.style.display = 'flex';
  await sleep(50);
  state.results.laplacian = applyLaplacian(gray, W, H);
  await sleep(80);

  // ── Stage 2: Sharpen
  setProgress(26, 'Sharpening image…', 'Sharpen');
  dom.loadingText.textContent = 'Sharpening with Laplacian…';
  await sleep(50);
  state.results.sharpened = applySharpen(gray, state.results.laplacian);
  await sleep(60);

  // ── Stage 3a: Sobel X
  setProgress(38, 'Sobel Gx (horizontal)…', 'Sobel-X');
  dom.loadingText.textContent = 'Detecting horizontal edges…';
  await sleep(50);
  state.results.sobelX = applySobelX(gray, W, H);
  await sleep(60);

  // ── Stage 3b: Sobel Y
  setProgress(48, 'Sobel Gy (vertical)…', 'Sobel-Y');
  dom.loadingText.textContent = 'Detecting vertical edges…';
  await sleep(50);
  state.results.sobelY = applySobelY(gray, W, H);
  await sleep(60);

  // ── Stage 3c: Magnitude
  setProgress(57, 'Computing Sobel magnitude…', 'Magnitude');
  dom.loadingText.textContent = 'Computing edge magnitude…';
  await sleep(50);
  state.results.sobel = sobelMagnitude(state.results.sobelX, state.results.sobelY);
  await sleep(60);

  // ── Stage 4: Averaging
  setProgress(67, 'Applying 5×5 averaging filter…', 'Average');
  dom.loadingText.textContent = 'Smoothing with 5×5 kernel…';
  await sleep(50);
  state.results.smoothed = applyAverage5x5(gray, W, H);
  await sleep(80);

  // ── Stage 5: Mask
  setProgress(77, 'Creating enhancement mask…', 'Mask');
  dom.loadingText.textContent = 'Building structural mask…';
  await sleep(50);
  state.results.mask = createMask(state.results.sobel, state.results.smoothed);
  await sleep(60);

  // ── Stage 6: Final composite
  setProgress(87, 'Compositing final image…', 'Composite');
  dom.loadingText.textContent = 'Combining original + mask…';
  await sleep(50);
  state.results.final = applyMaskToOriginal(gray, state.results.mask);
  await sleep(60);

  // ── Stage 7: Gamma
  setProgress(95, 'Applying gamma correction…', 'Gamma');
  dom.loadingText.textContent = 'Applying gamma curve…';
  await sleep(50);
  state.results.gamma = applyGamma(state.results.final, state.gammaValue);
  await sleep(60);

  setProgress(100, 'Pipeline complete!');

  // Hide loading
  await sleep(300);
  dom.loadingOverlay.style.display = 'none';

  // Render all stage cards
  renderPipelineGallery(W, H);

  // Draw comparison
  renderComparison(W, H);

  // Show sections
  dom.pipelineSection.style.display = 'block';
  dom.compareSection.style.display  = 'block';
  dom.downloadBtn.disabled = false;

  // Smooth scroll to pipeline
  setTimeout(() => {
    dom.pipelineSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 200);

  state.processing = false;
  dom.processBtn.disabled = false;
}

/* ════════════════════════════════════════════
   RENDER PIPELINE GALLERY
════════════════════════════════════════════ */
function renderPipelineGallery(W, H) {
  dom.stagesGrid.innerHTML = '';

  STAGES.forEach((stage, idx) => {
    const card = document.createElement('div');
    card.className = 'stage-card';
    card.style.animationDelay = (idx * 60) + 'ms';

    card.innerHTML = `
      <div class="stage-header">
        <span class="stage-num">0${idx + 1}</span>
        <span class="stage-title">${stage.title}</span>
        <span class="stage-tag">${stage.tag}</span>
      </div>
      <div class="stage-canvas-wrap">
        <canvas id="stageCanvas_${stage.key}"></canvas>
        <span class="stage-zoom-hint">↕ Hover to zoom</span>
      </div>
      <p class="stage-desc">${stage.desc}</p>
    `;

    dom.stagesGrid.appendChild(card);

    // Draw into canvas
    const canvas = document.getElementById(`stageCanvas_${stage.key}`);
    const data   = state.results[stage.key];
    if (!data) return;

    // Determine if normalisation needed
    const needsNorm = ['laplacian','sobelX','sobelY','sobel','mask'].includes(stage.key);
    drawGrayToCanvas(canvas, data, W, H, needsNorm);

    // Zoom on hover
    addZoomEffect(canvas);
  });

  // Update gamma stage desc
  const gammaDesc = dom.stagesGrid.querySelector('#stageCanvas_gamma')
    ?.closest('.stage-card')
    ?.querySelector('.stage-desc');
  if (gammaDesc) {
    gammaDesc.textContent = `s = c·rᵞ — power-law intensity mapping (γ = ${state.gammaValue.toFixed(2)}).`;
  }
}

/* Zoom effect on canvas hover */
function addZoomEffect(canvas) {
  const wrap = canvas.parentElement;
  wrap.style.cursor = 'zoom-in';

  wrap.addEventListener('mouseenter', () => {
    canvas.style.transition = 'transform 0.3s ease';
    canvas.style.transform  = 'scale(1.06)';
  });
  wrap.addEventListener('mouseleave', () => {
    canvas.style.transform = 'scale(1)';
  });
}

/* ════════════════════════════════════════════
   RENDER BEFORE / AFTER COMPARISON
════════════════════════════════════════════ */
function renderComparison(W, H) {
  // Draw original
  drawGrayToCanvas(dom.compareOriginal, state.results.original, W, H, false);

  // Draw final gamma-corrected
  drawGrayToCanvas(dom.compareFinal,   state.results.gamma,    W, H, false);

  // Set overlay canvas size
  dom.compareFinal.style.width  = '100%';
  dom.compareFinal.style.height = '100%';
  dom.compareOriginal.style.width  = '100%';

  // Init slider at 50%
  updateCompareSlider(0.5);
  initCompareSlider();
}

function updateCompareSlider(ratio) {
  const pct = ratio * 100;
  dom.compareFinal.style.clipPath   = `inset(0 ${100 - pct}% 0 0)`;
  dom.compareSlider.style.left      = pct + '%';
}

function initCompareSlider() {
  const container = dom.compareContainer;
  let dragging = false;

  const getX = (e) => {
    const rect = container.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  };

  container.addEventListener('mousedown',  (e) => { dragging = true; updateCompareSlider(getX(e)); });
  container.addEventListener('touchstart', (e) => { dragging = true; updateCompareSlider(getX(e)); }, { passive: true });
  window.addEventListener('mousemove',  (e) => { if (dragging) updateCompareSlider(getX(e)); });
  window.addEventListener('touchmove',  (e) => { if (dragging) updateCompareSlider(getX(e)); }, { passive: true });
  window.addEventListener('mouseup',  () => { dragging = false; });
  window.addEventListener('touchend', () => { dragging = false; });
}

/* ════════════════════════════════════════════
   GAMMA REAL-TIME UPDATE
════════════════════════════════════════════ */
function updateGamma(value) {
  state.gammaValue = value;
  dom.gammaValue.textContent = value.toFixed(2);

  // Update slider fill
  const pct = ((value - 0.1) / (3.0 - 0.1)) * 100;
  dom.gammaSlider.style.background =
    `linear-gradient(90deg, var(--cyan) ${pct}%, rgba(0,210,255,0.12) ${pct}%)`;

  // If pipeline has been run, re-apply gamma in real-time
  if (state.results.final) {
    state.results.gamma = applyGamma(state.results.final, value);

    // Update gamma card canvas
    const canvas = document.getElementById('stageCanvas_gamma');
    if (canvas) {
      drawGrayToCanvas(canvas, state.results.gamma, state.canvasW, state.canvasH, false);
    }

    // Update comparison overlay
    drawGrayToCanvas(dom.compareFinal, state.results.gamma, state.canvasW, state.canvasH, false);
  }
}

/* ════════════════════════════════════════════
   FILE UPLOAD
════════════════════════════════════════════ */
function handleImageFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    alert('Please upload a valid image file (PNG, JPG, TIFF, etc.)');
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      // Cap size for performance
      let W = img.naturalWidth;
      let H = img.naturalHeight;
      const MAX = 600;
      if (W > MAX || H > MAX) {
        const ratio = Math.min(MAX / W, MAX / H);
        W = Math.round(W * ratio);
        H = Math.round(H * ratio);
      }

      state.canvasW = W;
      state.canvasH = H;

      // Draw to hidden canvas, read pixels
      const tmpCanvas = document.createElement('canvas');
      tmpCanvas.width  = W;
      tmpCanvas.height = H;
      const tmpCtx = tmpCanvas.getContext('2d');
      tmpCtx.drawImage(img, 0, 0, W, H);
      state.originalImageData = tmpCtx.getImageData(0, 0, W, H);

      // Show original preview
      drawGrayToCanvas(dom.originalCanvas, imageDataToGray(state.originalImageData), W, H, false);

      // Image info
      dom.imageInfo.textContent = `${W} × ${H} px · ${(file.size / 1024).toFixed(1)} KB · ${file.type}`;

      // Draw histogram
      drawHistogram(imageDataToGray(state.originalImageData), dom.histogramCanvas);

      // Show control panel
      dom.controlPanel.style.display = 'block';
      dom.controlPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });

      // Reset pipeline output sections
      dom.pipelineSection.style.display = 'none';
      dom.compareSection.style.display  = 'none';
      dom.downloadBtn.disabled = true;
      state.results = {};
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

/* ════════════════════════════════════════════
   DOWNLOAD
════════════════════════════════════════════ */
function downloadResult() {
  if (!state.results.gamma) return;
  const canvas = document.createElement('canvas');
  canvas.width  = state.canvasW;
  canvas.height = state.canvasH;
  drawGrayToCanvas(canvas, state.results.gamma, state.canvasW, state.canvasH, false);

  const link = document.createElement('a');
  link.download = `medvision_enhanced_gamma${state.gammaValue.toFixed(2)}.png`;
  link.href     = canvas.toDataURL('image/png');
  link.click();
}

/* ════════════════════════════════════════════
   RESET
════════════════════════════════════════════ */
function reset() {
  state.originalImageData = null;
  state.results = {};
  state.canvasW = 0;
  state.canvasH = 0;

  dom.controlPanel.style.display    = 'none';
  dom.pipelineSection.style.display = 'none';
  dom.compareSection.style.display  = 'none';
  dom.downloadBtn.disabled = true;
  dom.progressWrap.style.display = 'none';
  dom.stagesGrid.innerHTML = '';
  dom.fileInput.value = '';

  // Reset gamma slider
  dom.gammaSlider.value = 1.0;
  updateGamma(1.0);

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ════════════════════════════════════════════
   EVENT LISTENERS
════════════════════════════════════════════ */

// File input
dom.fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) handleImageFile(e.target.files[0]);
});

// Upload zone click → trigger file input
dom.uploadZone.addEventListener('click', () => dom.fileInput.click());

// Drag and drop
dom.uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dom.uploadZone.classList.add('drag-over');
});
dom.uploadZone.addEventListener('dragleave', () => {
  dom.uploadZone.classList.remove('drag-over');
});
dom.uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dom.uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleImageFile(file);
});

// Gamma slider
dom.gammaSlider.addEventListener('input', (e) => updateGamma(parseFloat(e.target.value)));

// Process button
dom.processBtn.addEventListener('click', runPipeline);

// Reset button
dom.resetBtn.addEventListener('click', reset);

// Download button
dom.downloadBtn.addEventListener('click', downloadResult);

/* ════════════════════════════════════════════
   INIT
════════════════════════════════════════════ */
(function init() {
  // Initial slider style
  updateGamma(1.0);

  // Set histogram canvas responsive
  const histCanvas = dom.histogramCanvas;
  histCanvas.width  = 340;
  histCanvas.height = 160;

  // Animate hero stats in
  document.querySelectorAll('.stat').forEach((el, i) => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(16px)';
    setTimeout(() => {
      el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
      el.style.opacity    = '1';
      el.style.transform  = 'translateY(0)';
    }, 200 + i * 120);
  });

  console.log(
    '%cMedVision DIP v2.0 ready',
    'color:#00d2ff;font-family:monospace;font-size:14px;font-weight:bold'
  );
})();
