const API_BASE = 'http://localhost:8000';

let currentRepoUrl = '';
let currentRepoName = '';
let conversationHistory = [];

const repoUrlInput = document.getElementById('repoUrlInput');
const ingestBtn = document.getElementById('ingestBtn');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const chatContainer = document.getElementById('chatContainer');
const emptyState = document.getElementById('emptyState');
const statusPill = document.getElementById('statusPill');
const statusText = document.getElementById('statusText');

// Auto-detect GitHub URL from current tab
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab && tab.url) {
      const match = tab.url.match(/^https:\/\/github\.com\/([^\/]+\/[^\/]+)/);
      if (match) {
        currentRepoUrl = `https://github.com/${match[1]}`;
        repoUrlInput.value = currentRepoUrl;
      }
    }
  } catch (err) {
    console.error('Error auto-detecting tab URL:', err);
  }

  // Check health status of local backend
  checkBackendStatus();
});

async function checkBackendStatus() {
  try {
    const res = await fetch(`${API_BASE}/`);
    const data = await res.json();
    if (data.chain_ready && data.loaded_repo) {
      setLoadedState(data.loaded_repo);
    }
  } catch (err) {
    console.log('Backend not connected or no repo loaded yet.');
  }
}

function setLoadedState(repoName) {
  currentRepoName = repoName;
  statusPill.className = 'status-pill status-ready';
  statusText.textContent = repoName;
  queryInput.disabled = false;
  sendBtn.disabled = false;
  queryInput.placeholder = `Ask something about ${repoName}...`;
  if (emptyState) emptyState.style.display = 'none';
}

function setEmptyState() {
  statusPill.className = 'status-pill status-empty';
  statusText.textContent = 'No repo loaded';
  queryInput.disabled = true;
  sendBtn.disabled = true;
  queryInput.placeholder = 'Ask a question about this repository...';
}

// Ingest button handler
ingestBtn.addEventListener('click', async () => {
  const repoUrl = repoUrlInput.value.trim();
  if (!repoUrl) {
    alert('Please enter a GitHub repository URL.');
    return;
  }

  ingestBtn.disabled = true;
  ingestBtn.innerHTML = '<span class="spinner"></span> Ingesting...';

  try {
    const response = await fetch(`${API_BASE}/api/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_url: repoUrl })
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Failed to ingest repository');
    }

    conversationHistory = [];
    chatContainer.innerHTML = '';
    
    setLoadedState(data.repo_name);
    appendAssistantBubble(`Indexed **${data.chunk_count}** chunks from **${data.repo_name}**. Ask me anything about this codebase!`);
  } catch (err) {
    alert(`Ingestion Error: ${err.message}`);
    setEmptyState();
  } finally {
    ingestBtn.disabled = false;
    ingestBtn.textContent = 'Ingest';
  }
});

// Send Query Handler
sendBtn.addEventListener('click', handleSendQuery);
queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') handleSendQuery();
});

async function handleSendQuery() {
  const query = queryInput.value.trim();
  if (!query) return;

  if (emptyState) emptyState.style.display = 'none';

  appendUserBubble(query);
  queryInput.value = '';
  queryInput.disabled = true;
  sendBtn.disabled = true;

  const thinkingBubble = appendAssistantBubble('Thinking...', true);

  try {
    const response = await fetch(`${API_BASE}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: query,
        history: conversationHistory
      })
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Failed to query codebase');
    }

    // Remove thinking indicator
    thinkingBubble.remove();

    // Render Answer
    appendAssistantResponse(data, query);

    // Save to local conversation history
    conversationHistory.push([query, data.answer]);

  } catch (err) {
    thinkingBubble.textContent = `Error: ${err.message}`;
  } finally {
    queryInput.disabled = false;
    sendBtn.disabled = false;
    queryInput.focus();
  }
}

function appendUserBubble(text) {
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble user';
  bubble.textContent = text;
  chatContainer.appendChild(bubble);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function appendAssistantBubble(text, isTemp = false) {
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble assistant';
  bubble.textContent = text;
  chatContainer.appendChild(bubble);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  return bubble;
}

function appendAssistantResponse(data, originalQuery) {
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble assistant';

  // Standalone Rewrite Note
  if (data.standalone_question && data.standalone_question.trim() !== originalQuery.trim()) {
    const rewriteDiv = document.createElement('div');
    rewriteDiv.className = 'rewrite-note';
    rewriteDiv.textContent = `Interpreted as: "${data.standalone_question}"`;
    bubble.appendChild(rewriteDiv);
  }

  // Answer Text
  const answerDiv = document.createElement('div');
  answerDiv.style.whiteSpace = 'pre-wrap';
  answerDiv.textContent = data.answer;
  bubble.appendChild(answerDiv);

  // Sources / Citations
  if (data.sources && data.sources.length > 0) {
    const sourcesContainer = document.createElement('div');
    sourcesContainer.className = 'sources-container';

    data.sources.forEach(src => {
      const chip = document.createElement('a');
      chip.className = 'source-chip';
      chip.textContent = `${src.file} :: ${src.name}`;
      
      // Construct GitHub direct file link
      if (repoUrlInput.value) {
        const baseUrl = repoUrlInput.value.replace(/\/$/, '');
        chip.href = `${baseUrl}/blob/main/${src.file}`;
        chip.target = '_blank';
      }
      sourcesContainer.appendChild(chip);
    });

    bubble.appendChild(sourcesContainer);
  }

  chatContainer.appendChild(bubble);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}
