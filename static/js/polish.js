/**
 * MoodCast — Global polish pass.
 * Frontend-only finishing touches applied on every page: cursor spotlight
 * everywhere (previously landing-page-only), mouse parallax on the aurora
 * background, ripple on every button, gentle tilt on static panel cards,
 * a smooth fade-out page transition on internal navigation, and a
 * confetti burst other scripts can call when recommendations load. No
 * backend involvement — pure presentation layer, loaded last so it can
 * see DOM built by earlier page scripts.
 */

(function () {

  /* ---------------------------------------------------------------- */
  /* Cursor spotlight — inject on every page (index.html already has   */
  /* its own markup; this only adds one if it's missing).              */
  /* ---------------------------------------------------------------- */

  function ensureCursorSpotlight() {
    if (document.getElementById('cursorSpotlight')) {
      if (typeof initCursorSpotlight === 'function') {
        initCursorSpotlight(document.getElementById('cursorSpotlight'));
      }
      return;
    }
    if (typeof initCursorSpotlight !== 'function') return;
    // Relies on the shared `.cursor-spotlight` rule in design-system.css
    // (also used by index.html's own static markup for the same element).
    const el = document.createElement('div');
    el.className = 'cursor-spotlight';
    el.id = 'cursorSpotlight';
    document.body.appendChild(el);
    initCursorSpotlight(el);
  }

  /* ---------------------------------------------------------------- */
  /* Mouse parallax — nudges .aurora span via --px/--py custom props,   */
  /* composed with their existing float animation via `translate`.     */
  /* ---------------------------------------------------------------- */

  function initMouseParallax() {
    let raf = null, tx = 0, ty = 0;
    window.addEventListener('mousemove', (e) => {
      const nx = (e.clientX / window.innerWidth) * 2 - 1;
      const ny = (e.clientY / window.innerHeight) * 2 - 1;
      tx = nx; ty = ny;
      if (raf) return;
      raf = requestAnimationFrame(() => {
        document.documentElement.style.setProperty('--px', tx.toFixed(3));
        document.documentElement.style.setProperty('--py', ty.toFixed(3));
        raf = null;
      });
    }, { passive: true });
  }

  /* ---------------------------------------------------------------- */
  /* Global ripple — attaches to every button-like interactive element */
  /* found at load time, reusing landing-effects.js's attachRipple.    */
  /* ---------------------------------------------------------------- */

  function initGlobalRipple() {
    if (typeof attachRipple !== 'function') return;
    const selector = '.btn, .tab-btn, .mood-pill, .assistant-chip, .search-trigger, .assistant-orb, .track-action-btn, .poster-fav-btn, .np-play';
    document.querySelectorAll(selector).forEach((el) => {
      if (el.dataset.rippled) return;
      el.dataset.rippled = '1';
      el.classList.add('ripple-host');
      attachRipple(el);
    });
  }

  /* ---------------------------------------------------------------- */
  /* Gentle tilt on static/hero-style panels (not dense list rows,     */
  /* which already have their own hover treatment).                    */
  /* ---------------------------------------------------------------- */

  function initStaticTilt() {
    if (typeof attachTilt !== 'function') return;
    const selector = '.card-stat, .summary-card, .surprise-section, .insight-explain, .hero-panel, .hero-banner-content';
    document.querySelectorAll(selector).forEach((el) => {
      if (el.dataset.tilted) return;
      el.dataset.tilted = '1';
      attachTilt(el, { max: 4, scale: 1.015 });
    });
  }

  /* ---------------------------------------------------------------- */
  /* Page-leave transition on internal navigation.                     */
  /* ---------------------------------------------------------------- */

  function initPageTransitions() {
    document.addEventListener('click', (e) => {
      const a = e.target.closest('a[href]');
      if (!a) return;
      const href = a.getAttribute('href');
      if (!href || href.startsWith('#') || href.startsWith('http') || a.target === '_blank' || e.metaKey || e.ctrlKey) return;
      if (href.startsWith('/')) {
        e.preventDefault();
        document.body.classList.add('page-leaving');
        setTimeout(() => { window.location.href = href; }, 260);
      }
    });
  }

  /* ---------------------------------------------------------------- */
  /* Confetti — lightweight canvas burst, exposed as window.fireConfetti */
  /* so music.js / movies.js / dashboard.js can call it when real        */
  /* recommendations finish loading. Self-throttled so rapid mood       */
  /* switching doesn't spam it.                                         */
  /* ---------------------------------------------------------------- */

  let lastConfettiAt = 0;
  function fireConfetti(count = 90) {
    const now = Date.now();
    if (now - lastConfettiAt < 2500) return;
    lastConfettiAt = now;

    const canvas = document.createElement('canvas');
    canvas.id = 'confettiCanvas';
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    document.body.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    const moodA = getComputedStyle(document.documentElement).getPropertyValue('--mood-a').trim() || '#7f5af0';
    const moodB = getComputedStyle(document.documentElement).getPropertyValue('--mood-b').trim() || '#2cd4d9';
    const colors = [moodA, moodB, '#ffffff', '#f5c518'];

    const pieces = Array.from({ length: count }, () => ({
      x: canvas.width / 2 + (Math.random() - 0.5) * 160,
      y: canvas.height * 0.22 + (Math.random() - 0.5) * 60,
      vx: (Math.random() - 0.5) * 9,
      vy: -Math.random() * 7 - 3,
      size: Math.random() * 6 + 3,
      color: colors[Math.floor(Math.random() * colors.length)],
      rot: Math.random() * Math.PI,
      vr: (Math.random() - 0.5) * 0.3,
      shape: Math.random() < 0.5 ? 'rect' : 'circle',
    }));

    const gravity = 0.22;
    const start = performance.now();
    const duration = 1700;

    function tick(now2) {
      const elapsed = now2 - start;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const fade = Math.max(0, 1 - elapsed / duration);
      pieces.forEach((p) => {
        p.vy += gravity;
        p.x += p.vx;
        p.y += p.vy;
        p.rot += p.vr;
        ctx.save();
        ctx.globalAlpha = fade;
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rot);
        ctx.fillStyle = p.color;
        if (p.shape === 'rect') {
          ctx.fillRect(-p.size / 2, -p.size / 3, p.size, p.size * 0.6);
        } else {
          ctx.beginPath();
          ctx.arc(0, 0, p.size / 2, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      });
      if (elapsed < duration) {
        requestAnimationFrame(tick);
      } else {
        canvas.remove();
      }
    }
    requestAnimationFrame(tick);
  }
  window.fireConfetti = fireConfetti;

  /* ---------------------------------------------------------------- */
  /* Init                                                               */
  /* ---------------------------------------------------------------- */

  document.addEventListener('DOMContentLoaded', () => {
    ensureCursorSpotlight();
    initMouseParallax();
    initGlobalRipple();
    initStaticTilt();
    initPageTransitions();
    if (typeof onMoodThemeChange === 'function') {
      onMoodThemeChange(() => { initGlobalRipple(); initStaticTilt(); });
    }
  });
})();
