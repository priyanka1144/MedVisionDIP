'use strict';

const state = {
  originalImageData: null,   
  canvasW: 0,
  canvasH: 0,
  results: {},               
  gammaValue: 1.0,
  processing: false,
};

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

function drawGrayToCanvas(canvas, gray, width, height, doNormalise = false) {
  canvas.width  = width;
  canvas.height = height;
  const ctx  = canvas.getContext('2d');
  const data = doNormalise ? normalise(gray) : gray;
  ctx.putImageData(grayToImageData(data, width, height), 0, 0);
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

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

function applyLaplacian(gray, w, h) {
  const kernel = [
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0],
  ];
  return convolve(gray, w, h, kernel, false); 
}
function applySharpen(gray, laplacian) {
  const out = new Float32Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    out[i] = Math.max(0, Math.min(255, gray[i] + laplacian[i]));
  }
  return out;
}

function applySobelX(gray, w, h) {
  const Gx = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
  ];
  return convolve(gray, w, h, Gx, false);
}

function applySobelY(gray, w, h) {
  const Gy = [
    [ 1,  2,  1],
    [ 0,  0,  0],
    [-1, -2, -1],
  ];
  return convolve(gray, w, h, Gy, false);
}

function sobelMagnitude(gx, gy) {
  const out = new Float32Array(gx.length);
  for (let i = 0; i < gx.length; i++) {
    out[i] = Math.sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
  }
  return out;
}

function applyAverage5x5(gray, w, h) {
  const kernel = Array.from({ length: 5 }, () => Array(5).fill(1 / 25));
  return convolve(gray, w, h, kernel, true);
}

function createMask(sobelMag, smoothed) {
  const n1 = normalise(sobelMag);
  const n2 = normalise(smoothed);
  const out = new Float32Array(sobelMag.length);
  for (let i = 0; i < sobelMag.length; i++) {
    out[i] = (n1[i] / 255) * n2[i]; 
  }
  return out;
}

function applyMaskToOriginal(gray, mask) {
  const normMask = normalise(mask);
  const out = new Float32Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    out[i] = Math.max(0, Math.min(255, gray[i] + normMask[i] * 0.5));
  }
  return out;
}
function applyGamma(gray, gamma) {
  const out = new Float32Array(gray.length);
  const c   = 255;
  for (let i = 0; i < gray.length; i++) {
    out[i] = c * Math.pow(gray[i] / 255, gamma);
  }
  return out;
}

function drawHistogram(gray, canvas) {
  const ctx  = canvas.getContext('2d');
  const W    = canvas.width;
  const H    = canvas.height;
  ctx.clearRect(0, 0, W, H);


  const bins = new Array(256).fill(0);
  for (let i = 0; i < gray.length; i++) {
    bins[Math.round(Math.min(255, Math.max(0, gray[i])))]++;
  }
  const maxBin = Math.max(...bins);

  
  ctx.fillStyle = 'rgba(0,0,0,0.4)';
  ctx.fillRect(0, 0, W, H);

  
  ctx.strokeStyle = 'rgba(0,210,255,0.07)';
  ctx.lineWidth   = 1;
  for (let i = 1; i < 4; i++) {
    ctx.beginPath();
    ctx.moveTo(0, H * i / 4);
    ctx.lineTo(W, H * i / 4);
    ctx.stroke();
  }

  
  const barW = W / 256;
  for (let b = 0; b < 256; b++) {
    const barH  = (bins[b] / maxBin) * (H - 2);
    const ratio = b / 255;
    const r = Math.round(ratio * 0);
    const g = Math.round(ratio * 200 + 30);
    const blueC = 255;
    ctx.fillStyle = `rgba(${r},${g},${blueC},0.65)`;
    ctx.fillRect(b * barW, H - barH, barW, barH);
  }

  
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

async function runPipeline() {
  if (!state.originalImageData) return;
  if (state.processing) return;
  state.processing = true;
  dom.processBtn.disabled = true;
  dom.downloadBtn.disabled = true;

 
  dom.loadingText.textContent = 'Starting DIP pipeline…';
  dom.loadingOverlay.style.display = 'flex';

  initProgress();

  const { width: W, height: H } = state.originalImageData;
  await sleep(400);

 
  setProgress(5, 'Loading image pixels…', 'Load');
  const gray = imageDataToGray(state.originalImageData);
  state.results.original = gray;
  dom.loadingText.textContent = 'Applying Laplacian filter…';
  await sleep(200);

  
  setProgress(15, 'Computing Laplacian…', 'Laplacian');
  dom.loadingOverlay.style.display = 'flex';
  await sleep(50);
  state.results.laplacian = applyLaplacian(gray, W, H);
  await sleep(80);


  setProgress(26, 'Sharpening image…', 'Sharpen');
  dom.loadingText.textContent = 'Sharpening with Laplacian…';
  await sleep(50);
  state.results.sharpened = applySharpen(gray, state.results.laplacian);
  await sleep(60);


  setProgress(38, 'Sobel Gx (horizontal)…', 'Sobel-X');
  dom.loadingText.textContent = 'Detecting horizontal edges…';
  await sleep(50);
  state.results.sobelX = applySobelX(gray, W, H);
  await sleep(60);

 
  setProgress(48, 'Sobel Gy (vertical)…', 'Sobel-Y');
  dom.loadingText.textContent = 'Detecting vertical edges…';
  await sleep(50);
  state.results.sobelY = applySobelY(gray, W, H);
  await sleep(60);


  setProgress(57, 'Computing Sobel magnitude…', 'Magnitude');
  dom.loadingText.textContent = 'Computing edge magnitude…';
  await sleep(50);
  state.results.sobel = sobelMagnitude(state.results.sobelX, state.results.sobelY);
  await sleep(60);


  setProgress(67, 'Applying 5×5 averaging filter…', 'Average');
  dom.loadingText.textContent = 'Smoothing with 5×5 kernel…';
  await sleep(50);
  state.results.smoothed = applyAverage5x5(gray, W, H);
  await sleep(80);

 
  setProgress(77, 'Creating enhancement mask…', 'Mask');
  dom.loadingText.textContent = 'Building structural mask…';
  await sleep(50);
  state.results.mask = createMask(state.results.sobel, state.results.smoothed);
  await sleep(60);

  
  setProgress(87, 'Compositing final image…', 'Composite');
  dom.loadingText.textContent = 'Combining original + mask…';
  await sleep(50);
  state.results.final = applyMaskToOriginal(gray, state.results.mask);
  await sleep(60);

 
  setProgress(95, 'Applying gamma correction…', 'Gamma');
  dom.loadingText.textContent = 'Applying gamma curve…';
  await sleep(50);
  state.results.gamma = applyGamma(state.results.final, state.gammaValue);
  await sleep(60);

  setProgress(100, 'Pipeline complete!');


  await sleep(300);
  dom.loadingOverlay.style.display = 'none';


  renderPipelineGallery(W, H);


  renderComparison(W, H);

 
  dom.pipelineSection.style.display = 'block';
  dom.compareSection.style.display  = 'block';
  dom.downloadBtn.disabled = false;


  setTimeout(() => {
    dom.pipelineSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 200);

  state.processing = false;
  dom.processBtn.disabled = false;
}

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

   
    const canvas = document.getElementById(`stageCanvas_${stage.key}`);
    const data   = state.results[stage.key];
    if (!data) return;

    const needsNorm = ['laplacian','sobelX','sobelY','sobel','mask'].includes(stage.key);
    drawGrayToCanvas(canvas, data, W, H, needsNorm);

    addZoomEffect(canvas);
  });


  const gammaDesc = dom.stagesGrid.querySelector('#stageCanvas_gamma')
    ?.closest('.stage-card')
    ?.querySelector('.stage-desc');
  if (gammaDesc) {
    gammaDesc.textContent = `s = c·rᵞ — power-law intensity mapping (γ = ${state.gammaValue.toFixed(2)}).`;
  }
}

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

function renderComparison(W, H) {

  drawGrayToCanvas(dom.compareOriginal, state.results.original, W, H, false);

 
  drawGrayToCanvas(dom.compareFinal,   state.results.gamma,    W, H, false);


  dom.compareFinal.style.width  = '100%';
  dom.compareFinal.style.height = '100%';
  dom.compareOriginal.style.width  = '100%';

 
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
function updateGamma(value) {
  state.gammaValue = value;
  dom.gammaValue.textContent = value.toFixed(2);

  const pct = ((value - 0.1) / (3.0 - 0.1)) * 100;
  dom.gammaSlider.style.background =
    `linear-gradient(90deg, var(--cyan) ${pct}%, rgba(0,210,255,0.12) ${pct}%)`;

  if (state.results.final) {
    state.results.gamma = applyGamma(state.results.final, value);

    const canvas = document.getElementById('stageCanvas_gamma');
    if (canvas) {
      drawGrayToCanvas(canvas, state.results.gamma, state.canvasW, state.canvasH, false);
    }


    drawGrayToCanvas(dom.compareFinal, state.results.gamma, state.canvasW, state.canvasH, false);
  }
}


function handleImageFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    alert('Please upload a valid image file (PNG, JPG, TIFF, etc.)');
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      
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

    
      const tmpCanvas = document.createElement('canvas');
      tmpCanvas.width  = W;
      tmpCanvas.height = H;
      const tmpCtx = tmpCanvas.getContext('2d');
      tmpCtx.drawImage(img, 0, 0, W, H);
      state.originalImageData = tmpCtx.getImageData(0, 0, W, H);

    
      drawGrayToCanvas(dom.originalCanvas, imageDataToGray(state.originalImageData), W, H, false);

   
      dom.imageInfo.textContent = `${W} × ${H} px · ${(file.size / 1024).toFixed(1)} KB · ${file.type}`;

    
      drawHistogram(imageDataToGray(state.originalImageData), dom.histogramCanvas);

     
      dom.controlPanel.style.display = 'block';
      dom.controlPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });

   
      dom.pipelineSection.style.display = 'none';
      dom.compareSection.style.display  = 'none';
      dom.downloadBtn.disabled = true;
      state.results = {};
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

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

 
  dom.gammaSlider.value = 1.0;
  updateGamma(1.0);

  window.scrollTo({ top: 0, behavior: 'smooth' });
}




dom.fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) handleImageFile(e.target.files[0]);
});


dom.uploadZone.addEventListener('click', () => dom.fileInput.click());

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


dom.gammaSlider.addEventListener('input', (e) => updateGamma(parseFloat(e.target.value)));


dom.processBtn.addEventListener('click', runPipeline);


dom.resetBtn.addEventListener('click', reset);


dom.downloadBtn.addEventListener('click', downloadResult);


(function init() {
 
  updateGamma(1.0);

  
  const histCanvas = dom.histogramCanvas;
  histCanvas.width  = 340;
  histCanvas.height = 160;

 
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
