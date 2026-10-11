/**
 * MoodCast — AI Assistant.
 * Floating glass orb + chat panel, injected on every page that includes
 * this script. Frontend-only: wired exclusively to the two endpoints that
 * already exist —
 *   POST /api/detect-mood   { text } -> { mood, confidence, source }
 *   POST /api/recommend     { mood, mode, type, limit } -> { results, ... }
 * — plus the existing mood-theme.js engine for switching the live mood
 * theme. No new backend routes, no fabricated AI model; every "smart"
 * response is either a real classifier/ranking result from those two
 * endpoints, or a static explanation of how the real ranking algorithm
 * works (matches FEATURES.md).
 */

(function () {
  const CHAT_KEY = 'moodcast_assistant_chat';
  const MAX_HISTORY = 40;

  let panelOpen = false;
  let messages = loadHistory();

  function loadHistory() {
    try { return JSON.parse(sessionStorage.getItem(CHAT_KEY)) || []; }
    catch { return []; }
  }
  function saveHistory() {
    sessionStorage.setItem(CHAT_KEY, JSON.stringify(messages.slice(-MAX_HISTORY)));
  }

  // escapeHtml/showToast come from the shared static/js/ui-utils.js,
  // loaded before this script on every page.

  /* ---------------------------------------------------------------- */
  /* DOM injection                                                     */
  /* ---------------------------------------------------------------- */

  function injectDOM() {
    const orb = document.createElement('button');
    orb.className = 'assistant-orb';
    orb.id = 'assistantOrb';
    orb.setAttribute('aria-label', 'Open MoodCast assistant');
    orb.innerHTML = `✨<span class="assistant-orb-dot"></span>`;
    document.body.appendChild(orb);

    const panel = document.createElement('div');
    panel.className = 'assistant-panel glass-strong';
    panel.id = 'assistantPanel';
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-modal', 'true');
    panel.setAttribute('aria-label', 'MoodCast Assistant');
    panel.innerHTML = `
      <div class="assistant-header">
        <div class="assistant-header-orb" id="assistantHeaderOrb" aria-hidden="true">✨</div>
        <div class="assistant-header-text">
          <div class="assistant-header-title">MoodCast Assistant</div>
          <div class="assistant-header-sub" id="assistantHeaderSub">Here to help you find the vibe</div>
        </div>
        <button class="assistant-header-close" id="assistantClose" aria-label="Close assistant">✕</button>
      </div>
      <div class="assistant-messages" id="assistantMessages" aria-live="polite" role="log"></div>
      <div class="assistant-suggestions" id="assistantSuggestions"></div>
      <div class="assistant-input-row">
        <label for="assistantInput" class="visually-hidden">Describe how you're feeling</label>
        <input type="text" id="assistantInput" placeholder="Tell me how you're feeling…" autocomplete="off">
        <button class="assistant-send-btn" id="assistantSend" aria-label="Send message">➤</button>
      </div>
    `;
    document.body.appendChild(panel);

    orb.addEventListener('click', togglePanel);
    document.getElementById('assistantClose').addEventListener('click', () => setPanel(false));
    document.getElementById('assistantSend').addEventListener('click', handleUserInput);
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && panelOpen) setPanel(false);
    });
    document.getElementById('assistantInput').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') handleUserInput();
    });
  }

  function togglePanel() { setPanel(!panelOpen); }
  function setPanel(open) {
    panelOpen = open;
    document.getElementById('assistantOrb').classList.toggle('open', open);
    document.getElementById('assistantPanel').classList.toggle('open', open);
    if (open && !messages.length) {
      greet();
    }
    if (open) {
      renderSuggestions(defaultSuggestions());
      scrollToBottom();
    }
  }

  /* ---------------------------------------------------------------- */
  /* Message rendering                                                 */
  /* ---------------------------------------------------------------- */

  function scrollToBottom() {
    const box = document.getElementById('assistantMessages');
    box.scrollTop = box.scrollHeight;
  }

  function addBubble(role, html, { persist = true } = {}) {
    const box = document.getElementById('assistantMessages');
    const row = document.createElement('div');
    row.className = 'assistant-msg from-' + role;
    row.innerHTML = `
      <div class="assistant-msg-avatar">${role === 'bot' ? '✨' : '🙂'}</div>
      <div class="assistant-bubble ${role === 'bot' ? 'glass' : ''}">${html}</div>
    `;
    box.appendChild(row);
    scrollToBottom();
    if (persist) { messages.push({ role, html }); saveHistory(); }
    return row;
  }

  function showTyping() {
    const box = document.getElementById('assistantMessages');
    const row = document.createElement('div');
    row.className = 'assistant-msg from-bot';
    row.id = 'assistantTypingRow';
    row.innerHTML = `
      <div class="assistant-msg-avatar">✨</div>
      <div class="assistant-typing glass">
        <span class="dot-loader"><span></span><span></span><span></span></span>
      </div>
    `;
    box.appendChild(row);
    scrollToBottom();
  }
  function hideTyping() {
    const row = document.getElementById('assistantTypingRow');
    if (row) row.remove();
  }

  async function botSay(html, delay = 550) {
    showTyping();
    await new Promise((r) => setTimeout(r, delay));
    hideTyping();
    addBubble('bot', html);
  }

  function renderSuggestions(list) {
    const wrap = document.getElementById('assistantSuggestions');
    wrap.innerHTML = list.map((s) => `<button class="assistant-chip" data-action="${s.action}">${s.label}</button>`).join('');
    [...wrap.querySelectorAll('.assistant-chip')].forEach((btn, i) => {
      btn.addEventListener('click', () => runAction(list[i].action));
    });
  }

  function defaultSuggestions() {
    return [
      { action: 'happier-songs', label: '🎵 Recommend happier songs' },
      { action: 'calmer-movies', label: '🎬 Recommend calmer movies' },
      { action: 'switch-mood', label: '🔄 Switch mood' },
      { action: 'surprise-me', label: '🎲 Surprise me' },
      { action: 'explain', label: '💡 Explain recommendation' },
    ];
  }

  function greet() {
    const mood = getCurrentMood();
    const theme = getMoodTheme(mood);
    addBubble('bot', `Hey! I'm your MoodCast assistant. You're currently set to <strong>${theme.emoji} ${theme.label}</strong>. Want me to line something up, or just tell me how you're feeling?`);
  }

  /* ---------------------------------------------------------------- */
  /* Mini result cards (built from real /api/recommend results)        */
  /* ---------------------------------------------------------------- */

  async function fetchRecommend(mood, mode, type, limit = 5) {
    const res = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mood, mode, type, limit }),
    });
    return readJsonResponse(res);
  }

  function cardsHtml(results, type) {
    if (!results.length) return '<div style="opacity:.7">No results found for that combination.</div>';
    const Track = window.MoodcastTrack;
    const cards = results.map((r) => {
      // music: only real metadata (placeholders become "Track details unavailable"), linked to the exact track
      const info = type === 'movies' ? null : Track.trackDisplay(r);
      const title = escapeHtml(info ? info.heading : r.title);
      const img = type === 'movies' && r.image
        ? `<img class="mini-art" src="${escapeHtml(r.image)}" alt="" loading="lazy" decoding="async" onerror="this.outerHTML='<div class=&quot;mini-art-fallback&quot;>🎬</div>'">`
        : `<div class="mini-art-fallback">${type === 'movies' ? '🎬' : '🎵'}</div>`;
      const year = r.year && r.year !== 'N/A' ? r.year : '';
      const sub = type === 'movies' ? escapeHtml(year) : escapeHtml((info && info.artist) || '');
      const url = info ? Track.spotifyTrackUrl(r) : null;
      const body = `${img}<div class="mini-body"><div class="mini-title">${title}</div><div class="mini-sub">${sub}</div></div>`;
      return url
        ? `<a class="assistant-mini-card glass" href="${url}" target="_blank" rel="noopener noreferrer" aria-label="Open ${escapeHtml(info.title || 'this track')} on Spotify (new tab)">${body}</a>`
        : `<div class="assistant-mini-card glass">${body}</div>`;
    }).join('');
    return `<div class="assistant-cards-row">${cards}</div>`;
  }

  function pageLinkChip(type) {
    const href = type === 'movies' ? '/movies' : '/music';
    const label = type === 'movies' ? 'Open full Movies page →' : 'Open full Music page →';
    return `<div style="margin-top:8px"><a href="${href}" style="color:var(--mood-glow);font-weight:700;font-size:12px">${label}</a></div>`;
  }

  /* ---------------------------------------------------------------- */
  /* Suggestion actions — all backed by the two existing endpoints     */
  /* ---------------------------------------------------------------- */

  async function runAction(action) {
    addBubble('user', defaultSuggestions().concat([
      { action: 'match-mood', label: 'Match my mood' },
      { action: 'cheer-mood', label: 'Cheer me up' },
    ]).find((s) => s.action === action)?.label || action);
    renderSuggestions([]);

    try {
      await runActionBody(action);
    } catch (err) {
      reportRecommendError(err);
    }
    renderSuggestions(defaultSuggestions());
  }

  function reportRecommendError(err) {
    hideTyping();
    addBubble('bot', `Sorry — I couldn't load recommendations right now. Please try again in a moment.`);
    console.error(err);
  }

  async function runActionBody(action) {
    if (action === 'happier-songs') {
      showTyping();
      const mood = getCurrentMood();
      const data = await fetchRecommend(mood, 'uplift', 'music', 5);
      hideTyping();
      addBubble('bot', `Here's a lift for your <strong>${escapeHtml(mood)}</strong> mood — real picks from the cheer-up ranking:` + cardsHtml(data.results || [], 'music') + pageLinkChip('music'));
    } else if (action === 'calmer-movies') {
      showTyping();
      const data = await fetchRecommend('chill', 'match', 'movies', 5);
      hideTyping();
      addBubble('bot', `Some calm, chill-mood movies:` + cardsHtml(data.results || [], 'movies') + pageLinkChip('movies'));
    } else if (action === 'switch-mood') {
      await botSay('Which mood should I switch you to?', 350);
      const chipsHtml = MOOD_ORDER.map((id) => {
        const t = MOOD_THEME[id];
        return `<button class="assistant-chip" data-switch-mood="${id}">${t.emoji} ${t.label}</button>`;
      }).join('');
      addBubble('bot', `<div class="assistant-chip-row">${chipsHtml}</div>`, { persist: false });
      const box = document.getElementById('assistantMessages');
      [...box.querySelectorAll('[data-switch-mood]')].forEach((btn) => {
        btn.addEventListener('click', () => {
          const id = btn.dataset.switchMood;
          setMoodTheme(id);
          addBubble('user', MOOD_THEME[id].emoji + ' ' + MOOD_THEME[id].label);
          botSay(`Switched your mood theme to <strong>${MOOD_THEME[id].emoji} ${MOOD_THEME[id].label}</strong> — the whole page just re-themed. Ask me for recommendations any time.`);
          updateHeaderMood();
        }, { once: true });
      });
    } else if (action === 'surprise-me') {
      showTyping();
      const randomMood = MOOD_ORDER[Math.floor(Math.random() * MOOD_ORDER.length)];
      const randomType = Math.random() < 0.5 ? 'music' : 'movies';
      const randomMode = Math.random() < 0.5 ? 'match' : 'uplift';
      const data = await fetchRecommend(randomMood, randomMode, randomType, 5);
      hideTyping();
      const theme = getMoodTheme(randomMood);
      addBubble('bot', `🎲 Surprise pick: <strong>${theme.emoji} ${theme.label}</strong> ${randomMode === 'uplift' ? '(cheer-up mode)' : ''} ${randomType}:` + cardsHtml(data.results || [], randomType) + pageLinkChip(randomType));
    } else if (action === 'explain') {
      const mood = getCurrentMood();
      const mode = getCurrentMode();
      const conf = sessionStorage.getItem('moodConfidence');
      const confLine = conf !== null && !isNaN(parseFloat(conf))
        ? `Your last text analysis detected <strong>${Math.round(parseFloat(conf) * 100)}%</strong> confidence for this mood.`
        : `You selected this mood directly, so there's no ML confidence score for it — only text descriptions get a confidence score.`;
      await botSay(
        `Here's how recommendations actually work, no magic: for <strong>music</strong>, each mood maps to one or more target moods; tracks labelled with a target mood are ranked by an audio-feature classifier's probability for it (angry and stressed use audio-feature rules instead), and every target gets its share of the list. Popularity plays no part. For <strong>movies</strong>, it's a classifier's mood-probability from reading plot + genre text (65%), blended with rating (20%) and popularity (15%). ${confLine} You're currently in <strong>${mode === 'uplift' ? 'Cheer me up' : 'Match my mood'}</strong> mode.`,
        700
      );
    }
  }

  function updateHeaderMood() {
    const mood = getCurrentMood();
    const theme = getMoodTheme(mood);
    const orb = document.getElementById('assistantHeaderOrb');
    if (orb) orb.textContent = theme.emoji;
  }

  /* ---------------------------------------------------------------- */
  /* Free-text input -> real /api/detect-mood classifier               */
  /* ---------------------------------------------------------------- */

  async function handleUserInput() {
    const input = document.getElementById('assistantInput');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    addBubble('user', escapeHtml(text));
    renderSuggestions([]);
    showTyping();

    try {
      const res = await fetch('/api/detect-mood', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await readJsonResponse(res);
      hideTyping();

      if (data.error || !MOOD_THEME[data.mood]) {
        addBubble('bot', `I couldn't quite pick up a mood from that — try describing how you're feeling, or use a suggestion below.`);
      } else {
        const theme = getMoodTheme(data.mood);
        const pct = Math.round((data.confidence || 0) * 100);
        addBubble('bot', data.uncertain === true
          ? `I'm not sure about that one — my best guess is <strong>${theme.emoji} ${theme.label}</strong> (low confidence, ${pct}%). Want music or movies for that, or pick a different mood?`
          : `Sounds like <strong>${theme.emoji} ${theme.label}</strong> to me (${pct}% confidence). Want music or movies for that?`, { persist: true });
        const box = document.getElementById('assistantMessages');
        const rowId = 'mood-action-' + Date.now();
        addBubble('bot', `<div class="assistant-chip-row" id="${rowId}">
          <button class="assistant-chip" data-mm="music" data-mode="match">🎵 Match my mood</button>
          <button class="assistant-chip" data-mm="music" data-mode="uplift">🎵 Cheer me up</button>
          <button class="assistant-chip" data-mm="movies" data-mode="match">🎬 Match my mood</button>
          <button class="assistant-chip" data-mm="movies" data-mode="uplift">🎬 Cheer me up</button>
        </div>`, { persist: false });
        [...document.getElementById(rowId).querySelectorAll('[data-mm]')].forEach((btn) => {
          btn.addEventListener('click', async () => {
            const type = btn.dataset.mm, mode = btn.dataset.mode;
            setMoodTheme(data.mood, { confidence: data.confidence });
            addBubble('user', btn.textContent);
            showTyping();
            try {
              const rec = await fetchRecommend(data.mood, mode, type, 5);
              hideTyping();
              addBubble('bot', `Here you go:` + cardsHtml(rec.results || [], type) + pageLinkChip(type));
            } catch (err) {
              reportRecommendError(err);
            }
            renderSuggestions(defaultSuggestions());
            updateHeaderMood();
          }, { once: true });
        });
      }
    } catch (err) {
      hideTyping();
      addBubble('bot', `Something went wrong reaching the mood detector. Is the server running?`);
      console.error(err);
    }
    renderSuggestions(defaultSuggestions());
  }

  /* ---------------------------------------------------------------- */
  /* Init                                                               */
  /* ---------------------------------------------------------------- */

  document.addEventListener('DOMContentLoaded', () => {
    injectDOM();
    updateHeaderMood();
    // Rehydrate a persisted conversation (kept short + frontend-only).
    if (messages.length) {
      const box = document.getElementById('assistantMessages');
      messages.forEach((m) => {
        const row = document.createElement('div');
        row.className = 'assistant-msg from-' + m.role;
        row.innerHTML = `<div class="assistant-msg-avatar">${m.role === 'bot' ? '✨' : '🙂'}</div><div class="assistant-bubble ${m.role === 'bot' ? 'glass' : ''}">${m.html}</div>`;
        box.appendChild(row);
      });
    }
    if (typeof onMoodThemeChange === 'function') {
      onMoodThemeChange(updateHeaderMood);
    }
  });
})();
