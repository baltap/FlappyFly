// WebGL2 point-cloud view of every simulated neuron, placed at its soma (frontal view,
// fly's left on the right as if facing the fly). Spikes flash and decay.

export const CLASS_STYLE = {
  eye:     { col: [0.55, 0.78, 1.00], size: 1.6, label: 'Eye (lamina L1–L3)' },
  optic:   { col: [0.28, 0.45, 1.00], size: 1.4, label: 'Optic lobe' },
  vpn:     { col: [0.20, 0.80, 0.90], size: 1.8, label: 'Visual projection neurons' },
  central: { col: [0.62, 0.48, 1.00], size: 1.6, label: 'Central brain' },
  kc:      { col: [1.00, 0.80, 0.35], size: 2.4, label: 'Kenyon cells (memory)' },
  mbother: { col: [1.00, 0.80, 0.35], size: 4.0, label: 'APL / DPM' },
  mbon:    { col: [1.00, 0.58, 0.22], size: 4.2, label: 'MBONs (memory output)' },
  pam:     { col: [0.25, 0.90, 0.55], size: 3.2, label: 'PAM dopamine (reward)' },
  ppl1:    { col: [1.00, 0.33, 0.42], size: 4.2, label: 'PPL1 dopamine (punishment)' },
  dn:      { col: [0.92, 0.92, 0.95], size: 3.0, label: 'Descending neurons' },
  flap:    { col: [1.00, 0.38, 0.80], size: 6.0, label: 'DNg02 (wing power → flap)' },
};

const VS = `#version 300 es
in vec3 a_pos; in vec3 a_col; in float a_size; in float a_act;
uniform float u_yaw, u_pitch, u_scale, u_aspect, u_dpr, u_base;
out vec3 v_col; out float v_a;
void main() {
  float cy = cos(u_yaw), sy = sin(u_yaw), cp = cos(u_pitch), sp = sin(u_pitch);
  vec3 p = a_pos;
  vec3 q = vec3(cy * p.x + sy * p.z, p.y, -sy * p.x + cy * p.z);
  q = vec3(q.x, cp * q.y - sp * q.z, sp * q.y + cp * q.z);
  gl_Position = vec4(q.x * u_scale / u_aspect, q.y * u_scale, 0.0, 1.0);
  float a = clamp(a_act, 0.0, 1.0);
  gl_PointSize = a_size * u_dpr * (0.9 + 1.6 * a);
  v_col = a_col;
  v_a = u_base + a * 0.95;
}`;
const FS = `#version 300 es
precision mediump float;
in vec3 v_col; in float v_a; out vec4 o;
void main() {
  vec2 d = gl_PointCoord - 0.5; float r = dot(d, d);
  if (r > 0.25) discard;
  float f = exp(-r * 14.0);
  o = vec4(v_col * v_a * f, 1.0);
}`;

export class BrainView {
  constructor(canvas, init) {
    this.cv = canvas;
    const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: false, alpha: true });
    if (!gl) throw new Error('WebGL2 is not available');
    this.gl = gl;
    const n = init.n;
    this.n = n;
    const pam = new Set(init.roles.pam), ppl1 = new Set(init.roles.ppl1), flap = new Set(init.roles.flapDN);
    this.styleKey = new Array(n);
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), size = new Float32Array(n);
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < n; i++) { const p = init.pos[i]; cx += p[0]; cy += p[1]; cz += p[2]; }
    cx /= n; cy /= n; cz /= n;
    let ext = 0;
    for (let i = 0; i < n; i++) {
      const p = init.pos[i];
      let key = init.cls[i];
      if (key === 'dan') key = pam.has(i) ? 'pam' : 'ppl1';
      if (flap.has(i)) key = 'flap';
      this.styleKey[i] = key;
      const s = CLASS_STYLE[key] || CLASS_STYLE.central;
      pos[i * 3] = p[0] - cx; pos[i * 3 + 1] = -(p[1] - cy); pos[i * 3 + 2] = p[2] - cz;
      ext = Math.max(ext, Math.abs(pos[i * 3]), Math.abs(pos[i * 3 + 1]) * 1.6);
      col.set(s.col, i * 3);
      size[i] = s.size;
    }
    this.extent = ext;
    this.act = new Float32Array(n);
    this.boost = new Float32Array(n);
    const prog = this.prog = link(gl, VS, FS);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const buf = (data, name, k, usage = gl.STATIC_DRAW) => {
      const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, usage);
      const loc = gl.getAttribLocation(prog, name); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, k, gl.FLOAT, false, 0, 0);
      return b;
    };
    buf(pos, 'a_pos', 3); buf(col, 'a_col', 3); buf(size, 'a_size', 1);
    this.actBuf = buf(this.act, 'a_act', 1, gl.DYNAMIC_DRAW);
    this.u = {};
    for (const k of ['u_yaw', 'u_pitch', 'u_scale', 'u_aspect', 'u_dpr', 'u_base']) this.u[k] = gl.getUniformLocation(prog, k);
    this.yaw = 0; this.pitch = 0; this.autoYaw = 0;
    this._drag(canvas);
  }

  _drag(cv) {
    let down = null;
    cv.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY, this.yaw, this.pitch]; cv.setPointerCapture(e.pointerId); });
    cv.addEventListener('pointermove', (e) => {
      if (!down) return;
      this.yaw = down[2] + (e.clientX - down[0]) * 0.01;
      this.pitch = Math.max(-1.2, Math.min(1.2, down[3] + (e.clientY - down[1]) * 0.01));
    });
    const up = () => { down = null; };
    cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    cv.addEventListener('dblclick', () => { this.yaw = 0; this.pitch = 0; });
  }

  // counts: Uint8 spikes per neuron since the last frame
  update(counts, dtMs) {
    const a = this.act, decay = Math.exp(-dtMs / 90), boost = this.boost;
    for (let i = 0; i < this.n; i++) {
      const c = counts[i];
      let v = a[i] * decay;
      if (c) v = Math.min(1, v + 0.55 + 0.15 * c);
      if (boost[i] > 0) { v = Math.max(v, boost[i]); boost[i] *= 0.9; }
      a[i] = v;
    }
  }

  highlight(indices, level = 1) { for (const i of indices) this.boost[i] = Math.max(this.boost[i], level); }

  draw() {
    const gl = this.gl, cv = this.cv;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.round(cv.clientWidth * dpr), h = Math.round(cv.clientHeight * dpr);
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.actBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.act);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    const aspect = w / h;
    gl.uniform1f(this.u.u_yaw, this.yaw); gl.uniform1f(this.u.u_pitch, this.pitch);
    gl.uniform1f(this.u.u_scale, 0.94 / this.extent * Math.min(aspect, 1.7));
    gl.uniform1f(this.u.u_aspect, aspect); gl.uniform1f(this.u.u_dpr, dpr);
    gl.uniform1f(this.u.u_base, this.light ? 0.16 : 0.07);
    gl.drawArrays(gl.POINTS, 0, this.n);
  }
}

function link(gl, vs, fs) {
  const sh = (t, s) => { const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x); if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x)); return x; };
  const p = gl.createProgram();
  gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}
