/**
 * MoodCast — Landing page visual effects.
 * Frontend-only: particles canvas, cursor spotlight, 3D tilt on hover,
 * ripple on click. No backend/API involvement.
 */

function initParticles(canvas) {
  const ctx = canvas.getContext('2d');
  let W, H;
  function resize() { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }
  window.addEventListener('resize', resize);
  resize();

  const particles = Array.from({ length: 60 }, () => ({
    x: Math.random() * W,
    y: Math.random() * H,
    r: Math.random() * 1.7 + 0.5,
    speedY: Math.random() * 0.28 + 0.05,
    speedX: (Math.random() - 0.5) * 0.1,
    twinkle: Math.random() * Math.PI * 2,
  }));

  function getMoodGlow() {
    return getComputedStyle(document.documentElement).getPropertyValue('--mood-glow').trim() || '#7f5af0';
  }

  function tick() {
    ctx.clearRect(0, 0, W, H);
    const color = getMoodGlow();
    particles.forEach(p => {
      p.y -= p.speedY; p.x += p.speedX; p.twinkle += 0.018;
      if (p.y < -10) { p.y = H + 10; p.x = Math.random() * W; }
      if (p.x < -10) p.x = W + 10;
      if (p.x > W + 10) p.x = -10;
      const alpha = 0.25 + Math.sin(p.twinkle) * 0.2;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = Math.max(0, alpha);
      ctx.fill();
    });
    ctx.globalAlpha = 1;
    requestAnimationFrame(tick);
  }
  tick();
}

function initCursorSpotlight(el) {
  let targetX = window.innerWidth / 2, targetY = window.innerHeight * 0.3;
  let curX = targetX, curY = targetY;
  window.addEventListener('mousemove', e => { targetX = e.clientX; targetY = e.clientY; });
  function follow() {
    curX += (targetX - curX) * 0.07;
    curY += (targetY - curY) * 0.07;
    el.style.transform = `translate(${curX}px, ${curY}px) translate(-50%, -50%)`;
    requestAnimationFrame(follow);
  }
  follow();
}

/** 3D tilt: rotates a card toward the cursor position within its bounds. */
function attachTilt(card, { max = 10, scale = 1.045 } = {}) {
  let raf = null;
  function onMove(e) {
    const rect = card.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width;
    const py = (e.clientY - rect.top) / rect.height;
    const rotateY = (px - 0.5) * max * 2;
    const rotateX = (0.5 - py) * max * 2;
    if (raf) cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => {
      card.style.transform = `perspective(700px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale(${scale}) translateY(-8px)`;
      card.style.setProperty('--glow-x', (px * 100) + '%');
      card.style.setProperty('--glow-y', (py * 100) + '%');
    });
  }
  function onLeave() {
    if (raf) cancelAnimationFrame(raf);
    card.style.transform = '';
  }
  card.addEventListener('mousemove', onMove);
  card.addEventListener('mouseleave', onLeave);
}

/** Material-style ripple expanding from the click point. */
function attachRipple(el) {
  el.addEventListener('click', (e) => {
    const rect = el.getBoundingClientRect();
    const ripple = document.createElement('span');
    const size = Math.max(rect.width, rect.height) * 1.8;
    ripple.className = 'ripple-effect';
    ripple.style.width = ripple.style.height = size + 'px';
    ripple.style.left = (e.clientX - rect.left - size / 2) + 'px';
    ripple.style.top = (e.clientY - rect.top - size / 2) + 'px';
    el.appendChild(ripple);
    ripple.addEventListener('animationend', () => ripple.remove());
  });
}
