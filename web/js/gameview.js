// Canvas renderer for the game: a Flappy-style world flown by a fruit fly.

export class GameView {
  constructor(canvas) {
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.wing = 0;
    this.flash = 0;
    this.passGlow = 0;
  }

  draw(s, now) {
    const cv = this.cv, ctx = this.ctx;
    const o = s.o;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const k = Math.min(W / o.width, H / o.height);
    ctx.setTransform(dpr * k, 0, 0, dpr * k, dpr * (W - o.width * k) / 2, dpr * (H - o.height * k) / 2);
    // sky
    const sky = ctx.createLinearGradient(0, 0, 0, o.groundY);
    sky.addColorStop(0, '#7fd3e6'); sky.addColorStop(1, '#d9f3ee');
    ctx.fillStyle = sky; ctx.fillRect(0, 0, o.width, o.groundY);
    // distant skyline (parallax)
    ctx.fillStyle = 'rgba(120, 190, 190, 0.45)';
    const off = -(s.scroll * 0.15) % 64;
    for (let x = off - 64; x < o.width + 64; x += 32) {
      const hgt = 30 + ((Math.floor((x - off + s.scroll * 0.15) / 32) * 7919) % 5) * 9;
      ctx.fillRect(x, o.groundY - hgt, 28, hgt);
    }
    // pipes
    for (const [x, gy] of s.pipes) {
      const top = gy - o.gap / 2, bot = gy + o.gap / 2;
      this._pipe(ctx, x, 0, o.pipeW, top, true);
      this._pipe(ctx, x, bot, o.pipeW, o.groundY - bot, false);
    }
    // ground
    ctx.fillStyle = '#ded895'; ctx.fillRect(0, o.groundY, o.width, o.height - o.groundY);
    ctx.fillStyle = '#9bd35a'; ctx.fillRect(0, o.groundY, o.width, 12);
    ctx.fillStyle = '#7fbf45';
    const st = -(s.scroll % 24);
    for (let x = st - 24; x < o.width + 24; x += 24) {
      ctx.beginPath(); ctx.moveTo(x, o.groundY + 12); ctx.lineTo(x + 12, o.groundY + 12); ctx.lineTo(x + 20, o.groundY); ctx.lineTo(x + 8, o.groundY); ctx.fill();
    }
    ctx.fillStyle = '#c9c07a'; ctx.fillRect(0, o.groundY + 12, o.width, 3);
    // fly
    const sinceFlap = s.t - s.lastFlap;
    this.wing += 0.9;
    const tilt = Math.max(-0.5, Math.min(1.1, s.vy * 3.2));
    this._fly(ctx, o.birdX, s.y, tilt, sinceFlap < 160 ? 1 - sinceFlap / 160 : 0, s.dead);
    // flashes
    if (this.flash > 0.01) { ctx.fillStyle = `rgba(255,70,90,${this.flash * 0.45})`; ctx.fillRect(0, 0, o.width, o.height); this.flash *= 0.88; }
    if (this.passGlow > 0.01) { ctx.fillStyle = `rgba(80,255,160,${this.passGlow * 0.18})`; ctx.fillRect(0, 0, o.width, o.height); this.passGlow *= 0.9; }
    // score
    ctx.font = '700 44px ui-rounded, "SF Pro Rounded", system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.lineWidth = 6; ctx.strokeStyle = 'rgba(20,40,40,0.8)'; ctx.fillStyle = '#fff';
    ctx.strokeText(String(s.score), o.width / 2, 70); ctx.fillText(String(s.score), o.width / 2, 70);
  }

  _pipe(ctx, x, y, w, h, top) {
    if (h <= 0) return;
    const g = ctx.createLinearGradient(x, 0, x + w, 0);
    g.addColorStop(0, '#5aa832'); g.addColorStop(0.35, '#9be15d'); g.addColorStop(1, '#3f7f22');
    ctx.fillStyle = g; ctx.fillRect(x + 3, y, w - 6, h);
    const capH = 22, cy = top ? y + h - capH : y;
    ctx.fillRect(x - 1, cy, w + 2, capH);
    ctx.strokeStyle = '#2c5a18'; ctx.lineWidth = 2;
    ctx.strokeRect(x + 3, y, w - 6, h); ctx.strokeRect(x - 1, cy, w + 2, capH);
  }

  _fly(ctx, x, y, tilt, stroke, dead) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(tilt);
    // wings (flutter; big stroke right after a flap)
    const amp = 0.35 + stroke * 0.9;
    const a = Math.sin(this.wing) * amp;
    ctx.fillStyle = 'rgba(230,245,255,0.7)'; ctx.strokeStyle = 'rgba(120,140,160,0.9)'; ctx.lineWidth = 0.8;
    for (const s of [-1, 1]) {
      ctx.save(); ctx.rotate(-0.5 + s * 0.12 + a * (s > 0 ? 1 : 0.8));
      ctx.beginPath(); ctx.ellipse(-9, -7, 11, 4.2, -0.2, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); ctx.restore();
    }
    // abdomen
    ctx.fillStyle = '#c9a36a';
    ctx.beginPath(); ctx.ellipse(-8, 1, 8.5, 5.5, 0.1, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#4a3423';
    for (let i = 0; i < 3; i++) { ctx.beginPath(); ctx.ellipse(-12 + i * 3.2, 1, 1.1, 5, 0.1, 0, Math.PI * 2); ctx.fill(); }
    // thorax
    ctx.fillStyle = '#a8814f'; ctx.beginPath(); ctx.ellipse(1, -1, 6, 5, 0, 0, Math.PI * 2); ctx.fill();
    // head + big red eye
    ctx.fillStyle = '#8a6a40'; ctx.beginPath(); ctx.arc(7.5, -1.5, 4.2, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = dead ? '#5a1a1a' : '#d8262f'; ctx.beginPath(); ctx.ellipse(8.5, -2, 3.4, 3.8, 0.2, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.55)'; ctx.beginPath(); ctx.arc(9.4, -3.4, 1, 0, Math.PI * 2); ctx.fill();
    // legs
    ctx.strokeStyle = '#3b2a1a'; ctx.lineWidth = 0.9;
    for (let i = 0; i < 3; i++) { ctx.beginPath(); ctx.moveTo(-1 + i * 2.5, 3); ctx.lineTo(-3 + i * 3, 8); ctx.stroke(); }
    ctx.restore();
  }
}

// The fly's-eye view: every lamina column of the right eye as a hex dot at its
// viewing direction, grey level = photoreceptor signal.
export class EyeView {
  constructor(canvas, dirs, dirSide) {
    this.cv = canvas; this.ctx = canvas.getContext('2d'); this.dirs = dirs;
    this.show = dirs.map((_, d) => !dirSide || dirSide[d] === 'R');
  }
  draw(P) {
    const cv = this.cv, ctx = this.ctx;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const sx = W / 195, sy = H / 165, r = Math.max(1.6, Math.min(sx, sy) * 1.55);
    for (let d = 0; d < this.dirs.length; d++) {
      if (!this.show[d]) continue;
      const [az, el] = this.dirs[d];
      const v = Math.max(0, Math.min(255, P[d] * 255)) | 0;
      ctx.fillStyle = `rgb(${v * 0.92 | 0},${v | 0},${v * 0.85 | 0})`;
      ctx.beginPath(); ctx.arc((az + 22) * sx, (82 - el) * sy, r, 0, Math.PI * 2); ctx.fill();
    }
    ctx.strokeStyle = 'rgba(255,209,102,0.55)'; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(22 * sx, 0); ctx.lineTo(22 * sx, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, 82 * sy); ctx.lineTo(W, 82 * sy); ctx.stroke();
    ctx.setLineDash([]);
  }
}
