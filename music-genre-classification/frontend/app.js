const API_URL = 'http://localhost:8000';

const dropZone      = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const status        = document.getElementById('status');
const resultSection = document.getElementById('result-section');
const genreBadge    = document.getElementById('genre-badge');
const confidenceText= document.getElementById('confidence-text');
const chart         = document.getElementById('chart');
const resetBtn      = document.getElementById('reset-btn');

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) classifyFile(file);
});
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { if (fileInput.files[0]) classifyFile(fileInput.files[0]); });

async function classifyFile(file) {
  showStatus('Analyzing audio… extracting MFCC features…', 'loading');
  resultSection.classList.add('hidden');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      showStatus(`Error: ${err.detail}`, 'error');
      return;
    }
    const data = await res.json();
    renderResult(data);
  } catch (e) {
    showStatus('Cannot reach the server. Make sure the backend is running on port 8000.', 'error');
  }
}

function renderResult(data) {
  status.classList.add('hidden');
  genreBadge.textContent    = data.genre;
  confidenceText.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

  chart.innerHTML = '';
  data.scores.forEach((s, i) => {
    const pct = (s.score * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <span class="bar-label">${s.genre}</span>
      <div class="bar-track">
        <div class="bar-fill ${i === 0 ? 'top' : ''}" style="width: ${pct}%"></div>
      </div>
      <span class="bar-score">${pct}%</span>
    `;
    chart.appendChild(row);
  });

  resultSection.classList.remove('hidden');
}

function showStatus(msg, type) {
  status.textContent = msg;
  status.className = `status ${type}`;
}

resetBtn.addEventListener('click', () => {
  resultSection.classList.add('hidden');
  fileInput.value = '';
});
