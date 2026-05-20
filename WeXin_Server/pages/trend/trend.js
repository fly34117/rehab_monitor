const api = require('../../utils/api');
const app = getApp();

Page({
  data: {
    timeRange: 7,
    chartData: [],
    locked: false
  },

  onLoad() {
    this.setData({ locked: app.globalData.patientLocked });
    if (this.data.locked) this.loadData();
  },

  onShow() {
    const locked = app.globalData.patientLocked;
    if (locked !== this.data.locked) {
      this.setData({ locked });
      if (locked) this.loadData();
    }
  },

  onReady() {
    if (!this.data.locked) return;
    setTimeout(() => {
      this._initCanvases();
      this.drawCharts();
    }, 500);
  },

  _initCanvases() {
    const dpr = wx.getSystemInfoSync().pixelRatio;
    const query = wx.createSelectorQuery();

    query.select('#trendCanvas').fields({ node: true, size: true });
    query.select('#speedCanvas').fields({ node: true, size: true });

    query.exec((results) => {
      if (results[0]) {
        this.canvas = results[0].node;
        this.ctx = this.canvas.getContext('2d');
        this.canvasW = results[0].width;
        this.canvasH = results[0].height;
        this.canvas.width = this.canvasW * dpr;
        this.canvas.height = this.canvasH * dpr;
        this.ctx.scale(dpr, dpr);
      }
      if (results[1]) {
        this.speedCanvas = results[1].node;
        this.speedCtx = this.speedCanvas.getContext('2d');
        this.speedW = results[1].width;
        this.speedH = results[1].height;
        this.speedCanvas.width = this.speedW * dpr;
        this.speedCanvas.height = this.speedH * dpr;
        this.speedCtx.scale(dpr, dpr);
      }
    });
  },

  async loadData() {
    try {
      const data = await api.getHistoryTrend(this.data.timeRange);
      this.setData({ chartData: data });
      setTimeout(() => this.drawCharts(), 300);
    } catch (e) {
      wx.showToast({ title: '数据加载失败', icon: 'none' });
    }
  },

  switchRange(e) {
    this.setData({ timeRange: parseInt(e.currentTarget.dataset.days) });
    this.loadData();
  },

  drawCharts() {
    if (this.ctx && this.data.chartData.length) {
      this._drawLine(this.ctx, this.canvasW, this.canvasH,
        this.data.chartData, 'gait_rehab_score', '#1a73e8', [0, 100], 'GRS 评分');
    }
    if (this.speedCtx && this.data.chartData.length) {
      this._drawLine(this.speedCtx, this.speedW, this.speedH,
        this.data.chartData, 'gait_velocity_mps', '#27ae60', null, '步速 m/s');
    }
  },

  _drawLine(ctx, w, h, data, field, color, yRange, title) {
    ctx.clearRect(0, 0, w, h);
    if (data.length < 2) {
      ctx.fillStyle = '#ccc';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('数据不足', w / 2, h / 2);
      return;
    }

    const pad = { top: 16, right: 16, bottom: 28, left: 44 };
    const pw = w - pad.left - pad.right;
    const ph = h - pad.top - pad.bottom;

    const vals = data.map(d => d[field] || 0);
    const minY = yRange ? yRange[0] : Math.max(0, Math.min(...vals) * 0.85);
    const maxY = yRange ? yRange[1] : Math.max(...vals) * 1.12;
    const rangeY = Math.max(maxY - minY, 0.1);

    const toX = (i) => pad.left + (i / Math.max(data.length - 1, 1)) * pw;
    const toY = (v) => pad.top + ph - ((v - minY) / rangeY) * ph;

    ctx.fillStyle = '#aaa';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    const ySteps = 4;
    for (let j = 0; j <= ySteps; j++) {
      const val = minY + (rangeY / ySteps) * j;
      const y = toY(val);
      ctx.beginPath();
      ctx.strokeStyle = '#f0f0f0';
      ctx.lineWidth = 0.5;
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();
      ctx.fillText(val.toFixed(1), pad.left - 6, y + 4);
    }

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    data.forEach((p, i) => {
      const x = toX(i), y = toY(p[field] || 0);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    const lastX = toX(data.length - 1), baselineY = toY(minY);
    ctx.lineTo(lastX, baselineY);
    ctx.lineTo(pad.left, baselineY);
    ctx.closePath();
    const gradient = ctx.createLinearGradient(0, pad.top, 0, baselineY);
    gradient.addColorStop(0, color + '30');
    gradient.addColorStop(1, color + '00');
    ctx.fillStyle = gradient;
    ctx.fill();

    data.forEach((p, i) => {
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.arc(toX(i), toY(p[field] || 0), 3, 0, 2 * Math.PI);
      ctx.fill();
    });

    ctx.fillStyle = '#999';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    const step = Math.max(Math.floor(data.length / 5), 1);
    data.forEach((p, i) => {
      if (i % step === 0 || i === data.length - 1) {
        ctx.fillText((p.date || '').slice(5), toX(i), h - 6);
      }
    });
  },

  /** 接收 App 事件：后端确认锁定后重新加载数据 */
  onAppEvent(event) {
    if (event === 'backendLocked') {
      this.setData({ locked: true });
      this.loadData();
      setTimeout(() => {
        if (!this.canvas) this._initCanvases();
        setTimeout(() => this.drawCharts(), 300);
      }, 500);
    }
  }
});
