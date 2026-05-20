const api = require('../../utils/api');
const app = getApp();

Page({
  data: {
    currentPos: null,
    trajectory: [],
    locked: false
  },

  onLoad() {
    this.setData({ locked: app.globalData.patientLocked });
  },

  onShow() {
    const locked = app.globalData.patientLocked;
    if (locked !== this.data.locked) {
      this.setData({ locked });
      if (locked) this._startPolling();
      else this._stopPolling();
    }
  },

  onReady() {
    if (!this.data.locked) return;
    const query = wx.createSelectorQuery();
    query.select('#mapCanvas').fields({ node: true, size: true }).exec((res) => {
      if (res[0]) {
        const dpr = wx.getSystemInfoSync().pixelRatio;
        this.canvas = res[0].node;
        this.ctx = this.canvas.getContext('2d');
        this.cw = res[0].width;
        this.ch = res[0].height;
        this.canvas.width = this.cw * dpr;
        this.canvas.height = this.ch * dpr;
        this.ctx.scale(dpr, dpr);
      }
    });
    this._startPolling();
  },

  onUnload() {
    this._stopPolling();
  },

  _startPolling() {
    if (!app.globalData.patientLocked) return;
    this._stopPolling();
    this._fetch();
    this.timer = setInterval(() => this._fetch(), 3000);
  },

  _stopPolling() {
    clearInterval(this.timer);
    this.timer = null;
  },

  async _fetch() {
    try {
      const points = await api.getTrajectory(200);
      if (points && points.length > 0) {
        const last = points[points.length - 1];
        this.setData({
          trajectory: points,
          currentPos: { x: last.x.toFixed(2), y: last.y.toFixed(2) }
        });
        this._draw(points);
      }
    } catch (e) {}
  },

  _draw(points) {
    if (!this.ctx || !points.length) return;
    const ctx = this.ctx;
    const w = this.cw, h = this.ch;
    const pad = { top: 30, right: 20, bottom: 40, left: 50 };
    const pw = w - pad.left - pad.right;
    const ph = h - pad.top - pad.bottom;

    // 计算范围
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const minX = Math.min(...xs) - 0.5;
    const maxX = Math.max(...xs) + 0.5;
    const minY = Math.min(...ys) - 0.5;
    const maxY = Math.max(...ys) + 0.5;
    const rangeX = Math.max(maxX - minX, 2);
    const rangeY = Math.max(maxY - minY, 2);

    const toX = (v) => pad.left + ((v - minX) / rangeX) * pw;
    const toY = (v) => pad.top + ph - ((v - minY) / rangeY) * ph;

    ctx.clearRect(0, 0, w, h);

    // ---- 网格 ----
    ctx.strokeStyle = '#e8e8e8';
    ctx.lineWidth = 0.5;
    const stepX = rangeX > 10 ? 2 : 1;
    const stepY = rangeY > 10 ? 2 : 1;
    for (let gx = Math.floor(minX); gx <= Math.ceil(maxX); gx += stepX) {
      const x = toX(gx);
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + ph);
      ctx.stroke();
    }
    for (let gy = Math.floor(minY); gy <= Math.ceil(maxY); gy += stepY) {
      const y = toY(gy);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + pw, y);
      ctx.stroke();
    }

    // ---- 轴线 ----
    ctx.strokeStyle = '#bbb';
    ctx.lineWidth = 1;
    // Y轴 (X=0)
    if (minX <= 0 && maxX >= 0) {
      const x0 = toX(0);
      ctx.beginPath();
      ctx.moveTo(x0, pad.top);
      ctx.lineTo(x0, pad.top + ph);
      ctx.stroke();
    }
    // X轴 (Y=0)
    if (minY <= 0 && maxY >= 0) {
      const y0 = toY(0);
      ctx.beginPath();
      ctx.moveTo(pad.left, y0);
      ctx.lineTo(pad.left + pw, y0);
      ctx.stroke();
    }

    // ---- 轨迹线 ----
    ctx.beginPath();
    ctx.strokeStyle = '#1a73e8';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    points.forEach((p, i) => {
      const x = toX(p.x), y = toY(p.y);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // ---- 历史点 ----
    points.forEach((p, i) => {
      if (i % 3 !== 0) return;  // 每隔3个点画一个
      ctx.beginPath();
      ctx.fillStyle = '#95a5a6';
      ctx.arc(toX(p.x), toY(p.y), 2, 0, 2 * Math.PI);
      ctx.fill();
    });

    // ---- 当前位置 (大红点) ----
    const last = points[points.length - 1];
    const lx = toX(last.x), ly = toY(last.y);
    // 外圈光晕
    ctx.beginPath();
    ctx.arc(lx, ly, 10, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(231,76,60,0.2)';
    ctx.fill();
    // 实心红点
    ctx.beginPath();
    ctx.arc(lx, ly, 6, 0, 2 * Math.PI);
    ctx.fillStyle = '#e74c3c';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // ---- 坐标标注 ----
    ctx.fillStyle = '#999';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    for (let gx = Math.floor(minX); gx <= Math.ceil(maxX); gx += stepX) {
      ctx.fillText(gx.toString(), toX(gx), h - 6);
    }
    ctx.textAlign = 'right';
    for (let gy = Math.floor(minY); gy <= Math.ceil(maxY); gy += stepY) {
      ctx.fillText(gy.toString(), pad.left - 8, toY(gy) + 4);
    }
  },

  /** 接收 App 事件：后端确认锁定后重新加载数据 */
  onAppEvent(event) {
    if (event === 'backendLocked') {
      this.setData({ locked: true });
      this._startPolling();
    }
  }
});
