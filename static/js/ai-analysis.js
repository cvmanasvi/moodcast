/**
 * MoodCast — immersive AI analysis overlay.
 * Frontend-only presentation layer: builds a full-screen sequence of steps
 * with animated progress blocks, then fades out and calls onComplete().
 * Does not call any API or touch recommendation logic — purely a transition
 * experience between "Continue" and the next page.
 */

const AI_ANALYSIS_STEPS = [
  { icon: '\u{1F9E0}', label: 'Reading emotion' },
  { icon: '\u{1F3B5}', label: 'Analyzing audio features' },
  { icon: '\u{1F3AC}', label: 'Analyzing movie themes' },
  { icon: '✨', label: 'Generating recommendations' },
];

const AI_BLOCK_COUNT = 6;

function buildAIOverlay() {
  const overlay = document.createElement('div');
  overlay.className = 'ai-overlay';
  overlay.innerHTML = `
    <div class="ai-overlay-inner">
      <div class="ai-orb-wrap">
        <div class="ai-orb-ring"></div>
        <div class="ai-orb-ring ring-2"></div>
        <div class="ai-orb-core" id="aiOrbCore">🎭</div>
      </div>
      <div class="ai-title">Curating your perfect match</div>
      <div class="ai-subtitle">Hang tight — this only takes a moment.</div>
      <div class="ai-steps" id="aiSteps"></div>
      <div class="ai-estimate"><span class="dot-loader"><span></span><span></span><span></span></span> Estimated time: 2-3 seconds</div>
    </div>
  `;

  const stepsEl = overlay.querySelector('#aiSteps');
  AI_ANALYSIS_STEPS.forEach((step) => {
    const row = document.createElement('div');
    row.className = 'ai-step-row';
    const blocks = Array.from({ length: AI_BLOCK_COUNT }, () => '<span></span>').join('');
    row.innerHTML = `
      <div class="ai-step-icon"><span class="step-glyph">${step.icon}</span></div>
      <div class="ai-step-body">
        <div class="ai-step-label">${step.label}</div>
        <div class="ai-step-blocks">${blocks}</div>
      </div>
    `;
    stepsEl.appendChild(row);
  });

  return overlay;
}

/**
 * Runs the full 4-step sequence (~2.3s total, matches "2-3 seconds"),
 * fades in, steps through with staggered block-fill animations, fades out,
 * then calls options.onComplete(). Safe to call multiple times.
 */
function runAIAnalysis(options = {}) {
  const { onComplete = () => {}, mood = getCurrentMood?.() } = options;

  // The mood (and its confidence, if it came from text detection) was
  // already fully persisted by whatever step made Continue clickable, so
  // this only needs to (re-)apply the visual theme attribute — calling
  // setMoodTheme() again here would wipe that confidence, since it treats
  // a no-confidence call as a manual pick.
  if (mood) {
    document.documentElement.setAttribute('data-mood', mood);
  }

  const overlay = buildAIOverlay();
  document.body.appendChild(overlay);
  document.body.style.overflow = 'hidden';

  requestAnimationFrame(() => overlay.classList.add('show'));

  const rows = [...overlay.querySelectorAll('.ai-step-row')];
  const stepDuration = 560;

  function runStep(i) {
    if (i >= rows.length) {
      setTimeout(() => {
        overlay.classList.add('leaving');
        setTimeout(() => {
          overlay.remove();
          document.body.style.overflow = '';
          onComplete();
        }, 420);
      }, 260);
      return;
    }
    const row = rows[i];
    row.classList.add('active', 'filling');
    setTimeout(() => {
      row.classList.remove('active', 'filling');
      row.classList.add('done');
      runStep(i + 1);
    }, stepDuration);
  }

  runStep(0);
  return overlay;
}
