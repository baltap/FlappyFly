// Flappy Bird physics in continuous time (1 ms steps), original-game constants
// converted from per-frame (60 fps) units. Screen coordinates: y grows downward.

const F = 1000 / 60; // ms per original frame

export const GAME_DEFAULTS = {
  width: 288, height: 512, groundY: 400,
  birdX: 70, birdR: 10,
  gravity: 0.25 / (F * F),     // px/ms^2
  flapV: -4.6 / F,             // px/ms
  maxFall: 10 / F,             // px/ms
  pipeSpeed: 2 / F,            // px/ms
  pipeW: 52,
  gap: 100,                    // vertical opening
  spacing: 180,                // distance between pipe fronts
  firstPipe: 260,              // x of the first pipe front at start
  gapMargin: 60,               // min distance of a gap edge from top / ground
  maxGapShift: 140,            // max vertical change of the gap centre between pipes
  flapCooldown: 90,            // ms, fastest possible re-flap
};

export class Game {
  constructor(opts = {}, rng = Math.random) {
    this.o = { ...GAME_DEFAULTS, ...opts };
    this.rng = rng;
    this.reset();
  }

  reset() {
    const o = this.o;
    this.y = o.groundY * 0.45;
    this.vy = 0;
    this.t = 0;
    this.score = 0;
    this.dead = false;
    this.deathCause = null;
    this.lastFlap = -1e9;
    this.flaps = 0;
    this.scroll = 0; // total distance flown, for ground texture
    this.pipes = [];
    let x = o.firstPipe;
    let gy = o.groundY * 0.45;
    while (x < o.width + o.spacing * 2 || this.pipes.length < 3) {
      gy = this._nextGap(gy);
      this.pipes.push({ x, gapY: gy, passed: false });
      x += o.spacing;
    }
    this.events = [];
  }

  _nextGap(prev) {
    const o = this.o;
    const lo = o.gapMargin + o.gap / 2, hi = o.groundY - o.gapMargin - o.gap / 2;
    let g = prev + (this.rng() * 2 - 1) * o.maxGapShift;
    return Math.min(hi, Math.max(lo, g));
  }

  flap() {
    if (this.dead || this.t - this.lastFlap < this.o.flapCooldown) return false;
    this.vy = this.o.flapV;
    this.lastFlap = this.t;
    this.flaps++;
    return true;
  }

  // Advance dt ms. Returns events: 'pass', 'crash'.
  step(dt = 1) {
    const o = this.o;
    this.events.length = 0;
    if (this.dead) return this.events;
    this.t += dt;
    this.vy = Math.min(o.maxFall, this.vy + o.gravity * dt);
    this.y += this.vy * dt;
    const dx = o.pipeSpeed * dt;
    this.scroll += dx;
    for (const p of this.pipes) {
      p.x -= dx;
      if (!p.passed && p.x + o.pipeW < o.birdX - o.birdR) {
        p.passed = true;
        this.score++;
        this.events.push('pass');
      }
    }
    if (this.pipes[0].x + o.pipeW < -60) {
      this.pipes.shift();
      const last = this.pipes[this.pipes.length - 1];
      this.pipes.push({ x: last.x + o.spacing, gapY: this._nextGap(last.gapY), passed: false });
    }
    // collisions
    let cause = null;
    if (this.y + o.birdR >= o.groundY) cause = 'ground';
    else if (this.y - o.birdR <= 0) cause = 'ceiling';
    else {
      for (const p of this.pipes) {
        if (o.birdX + o.birdR > p.x && o.birdX - o.birdR < p.x + o.pipeW) {
          const top = p.gapY - o.gap / 2, bot = p.gapY + o.gap / 2;
          // circle vs pipe rectangles (approximate: vertical clearance in the pipe span)
          if (this.y - o.birdR * 0.8 < top) { cause = 'pipe-top'; break; }
          if (this.y + o.birdR * 0.8 > bot) { cause = 'pipe-bottom'; break; }
        }
      }
    }
    if (cause) {
      this.dead = true;
      this.deathCause = cause;
      this.events.push('crash');
    }
    return this.events;
  }

  // The next pipe the bird still has to clear.
  nextPipe() {
    const o = this.o;
    for (const p of this.pipes) if (p.x + o.pipeW >= o.birdX - o.birdR) return p;
    return this.pipes[0];
  }
}
