/* UMAY çekirdek: ön işleme, çözme (decode), NMS ve konuşma cümlesi mantığı.
   Hem tarayıcıda (window.UmayCore) hem Node'da (testler) çalışır. */
(function (g) {
  'use strict';

  /* YOLOX ham çıktısını çöz: [N, 5+C] → kutular. strides: [8,16,32] */
  function makeGrids(size, strides) {
    const gx = [], gy = [], st = [];
    for (const s of strides) {
      const n = Math.floor(size / s);
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { gx.push(x); gy.push(y); st.push(s); }
    }
    return { gx: Int16Array.from(gx), gy: Int16Array.from(gy), st: Int16Array.from(st), n: gx.length };
  }

  /* data: Float32Array [N*(5+C)]; döner: [{x1,y1,x2,y2,score,cls}] (model giriş uzayında, piksel) */
  function decode(data, grids, numClasses, scoreThr, rawDecode) {
    const stride = 5 + numClasses, out = [];
    for (let i = 0; i < grids.n; i++) {
      const o = i * stride, obj = data[o + 4];
      if (obj < scoreThr) continue;            // obj*cls <= obj olduğundan güvenli ön eleme
      let best = -1, bs = 0;
      for (let c = 0; c < numClasses; c++) { const v = data[o + 5 + c]; if (v > bs) { bs = v; best = c; } }
      const score = obj * bs;
      if (score < scoreThr) continue;
      let cx, cy, w, h;
      if (rawDecode) {
        const s = grids.st[i];
        cx = (data[o] + grids.gx[i]) * s; cy = (data[o + 1] + grids.gy[i]) * s;
        w = Math.exp(data[o + 2]) * s;    h = Math.exp(data[o + 3]) * s;
      } else { cx = data[o]; cy = data[o + 1]; w = data[o + 2]; h = data[o + 3]; }
      out.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score, cls: best });
    }
    return out;
  }

  function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1), x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return ua > 0 ? inter / ua : 0;
  }

  /* sınıf bazlı NMS */
  function nms(boxes, iouThr, maxOut) {
    boxes.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const b of boxes) {
      let ok = true;
      for (const k of keep) { if (k.cls === b.cls && iou(k, b) > iouThr) { ok = false; break; } }
      if (ok) { keep.push(b); if (keep.length >= (maxOut || 50)) break; }
    }
    return keep;
  }

  /* model uzayından (letterbox) kare uzayına: ölçek r, sol-üst yerleşim */
  function toFrame(boxes, r, fw, fh) {
    return boxes.map(b => ({
      x1: Math.max(0, b.x1 / r), y1: Math.max(0, b.y1 / r),
      x2: Math.min(fw, b.x2 / r), y2: Math.min(fh, b.y2 / r),
      score: b.score, cls: b.cls
    })).filter(b => b.x2 - b.x1 > 2 && b.y2 - b.y1 > 2);
  }

  /* yön ve yakınlık */
  function direction(b, fw) {
    const c = (b.x1 + b.x2) / 2 / fw;
    return c < 0.34 ? 'sol' : c > 0.66 ? 'sag' : 'on';
  }
  function nearness(b, fw, fh) {           // 0..1+ ; büyük = yakın
    const hr = (b.y2 - b.y1) / fh, ar = ((b.x2 - b.x1) * (b.y2 - b.y1)) / (fw * fh);
    return Math.max(hr * 0.8, Math.sqrt(ar));
  }

  /* Trafik ışığı rengi: kutunun içindeki parlak ve doygun pikselleri sayar.
     d: RGBA dizisi (kırpılmış kutu), w,h: kırpmanın boyutu. Döner: {color:'kirmizi'|'sari'|'yesil'|null, conf} */
  function lightColor(d, w, h) {
    if (w < 4 || h < 4) return { color: null, conf: 0 };
    const x0 = Math.floor(w * 0.08), x1 = Math.ceil(w * 0.92), y0 = Math.floor(h * 0.04), y1 = Math.ceil(h * 0.96);
    const vertical = h >= w * 1.5, K = ['r', 'y', 'g'];
    const sc = { r: 0, y: 0, g: 0 }, EXP = { r: 0, y: 1, g: 2 };
    for (let y = y0; y < y1; y++) {
      const band = Math.min(2, Math.floor((y - y0) / (y1 - y0) * 3));
      for (let x = x0; x < x1; x++) {
        const i = (y * w + x) * 4, r = d[i], g = d[i + 1], b = d[i + 2];
        const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
        if (mx < 200 || (mx - mn) < 0.2 * mx) continue;          // parlak ve doygun değilse lamba değil
        const dl = mx - mn; let hh;
        if (mx === r) hh = 60 * (((g - b) / dl) % 6); else if (mx === g) hh = 60 * ((b - r) / dl + 2); else hh = 60 * ((r - g) / dl + 4);
        if (hh < 0) hh += 360;
        let k = null;
        if (hh <= 20 || hh >= 340) k = 'r'; else if (hh >= 35 && hh <= 62) k = 'y'; else if (hh >= 90 && hh <= 190) k = 'g';
        if (!k) continue;
        let wgt = 1;
        if (vertical) wgt = band === EXP[k] ? 1.4 : (Math.abs(band - EXP[k]) === 2 ? 0.5 : 0.9);   // üstte kırmızı, altta yeşil beklenir
        sc[k] += wgt;
      }
    }
    const arr = K.map(k => [k, sc[k]]).sort((a, b) => b[1] - a[1]);
    const top = arr[0], sec = arr[1][1], area = (x1 - x0) * (y1 - y0);
    if (top[1] < Math.max(3, area * 0.004) || top[1] < 1.8 * sec) return { color: null, conf: 0 };
    return { color: { r: 'kirmizi', y: 'sari', g: 'yesil' }[top[0]], conf: top[1] / (top[1] + sec) };
  }
  const COLOR_TR = { kirmizi: 'kırmızı', sari: 'sarı', yesil: 'yeşil' };
  const DIR_TR = { sol: 'solunuzda', sag: 'sağınızda', on: 'önünüzde' };

  /* Ham algılamalardan konuşulacak adayları üret.
     classes: [{tr, say, w}]  (w: öncelik çarpanı) */
  function candidates(dets, classes, fw, fh, nearThr) {
    const list = [];
    for (const d of dets) {
      const cl = classes[d.cls]; if (!cl || !cl.say) continue;
      const dir = direction(d, fw), near = nearness(d, fw, fh);
      list.push({
        key: d.cls + ':' + dir, cls: d.cls, dir, near,
        isNear: near >= nearThr,
        prio: (cl.w || 1) * (near + 0.15) * (dir === 'on' ? 1.25 : 1),
        name: cl.tr, score: d.score, colorable: !!cl.color, rawColor: d.color || null, color: null
      });
    }
    // aynı anahtardan en yakını kalsın
    const best = new Map();
    for (const c of list) { const o = best.get(c.key); if (!o || c.prio > o.prio) best.set(c.key, c); }
    return [...best.values()].sort((a, b) => b.prio - a.prio);
  }

  function phrase(c) {
    const where = DIR_TR[c.dir];
    const nm = c.colorable ? (c.color ? COLOR_TR[c.color] + ' ' + c.name : c.name) : c.name;
    const tail = c.colorable && !c.color ? (c.isNear ? ', rengi anlaşılmıyor' : ' var, rengi anlaşılmıyor') : (c.isNear ? '' : ' var');
    return (c.isNear ? 'Dikkat, ' + where + ' yakın ' : where + ' ') + nm + tail;
  }

  /* Konuşma zamanlayıcı: titremeyi önler (2/3 kare), tekrar süresini uygular */
  function Speaker(opts) {
    this.repeatMs = opts.repeatMs; this.hist = new Map(); this.lastSaid = new Map(); this.lastNear = new Map(); this.colHist = new Map(); this.lastColor = new Map();
  }
  Speaker.prototype.update = function (cands, now) {
    const seenKeys = new Set(cands.map(c => c.key));
    for (const c of cands) {
      if (c.colorable) {                                   // son 5 karenin çoğunluk rengi (en az 3)
        const ch = this.colHist.get(c.key) || []; ch.push(c.rawColor); if (ch.length > 5) ch.shift(); this.colHist.set(c.key, ch);
        const cnt = {}; for (const v of ch) if (v) cnt[v] = (cnt[v] || 0) + 1;
        const bestC = Object.keys(cnt).sort((a, b) => cnt[b] - cnt[a])[0];
        c.color = bestC && cnt[bestC] >= 3 ? bestC : null;
      }
      const h = this.hist.get(c.key) || [];
      h.push(1); if (h.length > 3) h.shift(); this.hist.set(c.key, h);
    }
    for (const [k, h] of this.hist) { if (!seenKeys.has(k)) { h.push(0); if (h.length > 3) h.shift(); if (h.every(v => !v)) { this.hist.delete(k); this.colHist.delete(k); } } }
    const out = [];
    for (const c of cands) {
      const h = this.hist.get(c.key) || [];
      const stable = h.reduce((a, b) => a + b, 0) >= 2;
      if (!stable) continue;
      if (c.colorable && !c.color && (this.colHist.get(c.key) || []).length < 5) continue;   // rengi anlaşılana kadar kısa bekle
      const last = this.lastSaid.get(c.key) || -1e9;
      const wasNear = this.lastNear.get(c.key) || false;
      const becameNear = c.isNear && !wasNear;
      const rep = c.isNear ? Math.min(this.repeatMs, 3500) : this.repeatMs;
      const colorChanged = c.colorable && c.color && this.lastColor.has(c.key) && this.lastColor.get(c.key) !== c.color;
      if (becameNear || colorChanged || now - last >= rep) out.push(c);
    }
    return out;
  };
  Speaker.prototype.markSaid = function (c, now) { this.lastSaid.set(c.key, now); this.lastNear.set(c.key, c.isNear); if (c.colorable) this.lastColor.set(c.key, c.color); };
  Speaker.prototype.markNotNear = function (cands) {
    const nearKeys = new Set(cands.filter(c => c.isNear).map(c => c.key));
    for (const k of this.lastNear.keys()) if (!nearKeys.has(k)) this.lastNear.set(k, false);
  };

  g.UmayCore = { makeGrids, decode, iou, nms, toFrame, direction, nearness, candidates, phrase, Speaker, lightColor };
  if (typeof module !== 'undefined' && module.exports) module.exports = g.UmayCore;
})(typeof window !== 'undefined' ? window : globalThis);
