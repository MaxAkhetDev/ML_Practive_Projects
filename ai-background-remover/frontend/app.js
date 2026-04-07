const API_URL = 'http://localhost:8000';

const dropZone      = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const status        = document.getElementById('status');
const resultSection = document.getElementById('result-section');
const originalImg   = document.getElementById('original-preview');
const resultImg     = document.getElementById('result-preview');
const downloadBtn   = document.getElementById('download-btn');
const resetBtn      = document.getElementById('reset-btn');

let resultBlob = null;

// Drag-and-drop
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) processFile(file);
});

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { if (fileInput.files[0]) processFile(fileInput.files[0]); });

async function processFile(file) {
  originalImg.src = URL.createObjectURL(file);
  showStatus('Removing background… this may take a few seconds.', 'loading');
  resultSection.classList.add('hidden');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_URL}/remove-bg`, { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      showStatus(`Error: ${err.detail}`, 'error');
      return;
    }
    resultBlob = await res.blob();
    resultImg.src = URL.createObjectURL(resultBlob);
    status.classList.add('hidden');
    resultSection.classList.remove('hidden');
  } catch (e) {
    showStatus('Could not reach the server. Make sure the backend is running on port 8000.', 'error');
  }
}

function showStatus(msg, type) {
  status.textContent = msg;
  status.className = `status ${type}`;
}

downloadBtn.addEventListener('click', () => {
  if (!resultBlob) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(resultBlob);
  a.download = 'background-removed.png';
  a.click();
});

resetBtn.addEventListener('click', () => {
  resultSection.classList.add('hidden');
  fileInput.value = '';
  resultBlob = null;
});