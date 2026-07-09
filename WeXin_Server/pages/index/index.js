const api = require('../../utils/api');
const app = getApp();

Page({
  data: {
    locked: false,
    patientName: '',
    grsScore: 0,
    metricsList: [],
    wsConnected: false
  },

  _polling: false,

  onLoad() {
    this._refreshLockState();
    this._statusTimer = setInterval(() => {
      this.setData({ wsConnected: app.globalData.wsConnected });
    }, 2000);
  },

  onReady() {
    setTimeout(() => this._drawGauge(0), 300);
  },

  onShow() {
    const wasLocked = this.data.locked;
    this._refreshLockState();
    const nowLocked = app.globalData.patientLocked;
    if (!wasLocked && nowLocked) {
      this._setupWebSocket();
      setTimeout(() => this._startPolling(), 500);
      setTimeout(() => this._loadEmotionStats(), 1500);
    }
    if (wasLocked && !nowLocked) {
      this._stopPolling();
      this.setData({ grsScore: 0, metricsList: [] });
    }
  },

  _refreshLockState() {
    const locked = app.globalData.patientLocked;
    this.setData({
      locked: locked,
      patientName: app.globalData.patientName || app.globalData.patientId || '',
      wsConnected: app.globalData.wsConnected
    });
    if (locked && !this._polling) {
      this._setupWebSocket();
      setTimeout(() => this._startPolling(), 500);
      setTimeout(() => this._loadEmotionStats(), 1500);
    }
  },

  // ===== 数据获取 =====

  _startPolling() {
    if (this._polling) return;
    this._polling = true;
    this.pollTimer = setInterval(async () => {
      try {
        const metrics = await api.getRealtimeMetrics();
        this._updateMetrics(metrics);
      } catch (e) {
        // WebSocket 可用时静默失败
      }
    }, 3000);
  },

  _setupWebSocket() {
    if (!app.socket) app._initWebSocket();
    if (!app.socket) return;
    app.socket.subscribe((msg) => {
      if (msg.type === 'metrics') this._updateMetrics(msg.data);
    });
  },

  async _loadEmotionStats() {
    try {
      const stats = await api.getEmotionStats();
      this._drawEmotionChart(stats);
    } catch (e) {}
  },

  // ===== 指标更新 & 绘制 =====

  _updateMetrics(metrics) {
    if (!metrics) return;
    // 后端未锁定时清空数据显示
    if (!metrics._locked) {
      if (this.data.grsScore !== 0) {
        this.setData({ grsScore: 0, metricsList: [] });
        this._drawGauge(0);
      }
      return;
    }
    const list = [
      { label: '步速', value: (metrics.gait_velocity_mps || 0).toFixed(2), unit: 'm/s',
        status: (metrics.gait_velocity_mps || 0) > 0.8 ? 'good' : 'warn' },
      { label: '步长', value: (metrics.stride_length_m || 0).toFixed(2), unit: 'm',
        status: (metrics.stride_length_m || 0) > 1.0 ? 'good' : 'warn' },
      { label: '对称性', value: ((metrics.symmetry || 0) * 100).toFixed(0), unit: '%',
        status: (metrics.symmetry || 0) > 0.85 ? 'good' : 'warn' },
      { label: '步频', value: (metrics.cadence_spm || 0).toFixed(0), unit: '步/分',
        status: (metrics.cadence_spm || 0) > 70 ? 'good' : 'warn' },
      { label: '左膝ROM', value: (metrics.left_knee_rom || 0).toFixed(0), unit: '°',
        status: (metrics.left_knee_rom || 0) > 40 ? 'good' : 'warn' },
      { label: '右膝ROM', value: (metrics.right_knee_rom || 0).toFixed(0), unit: '°',
        status: (metrics.right_knee_rom || 0) > 40 ? 'good' : 'warn' },
      { label: '稳定度', value: (metrics.step_time_cv || 99) < 15 ? '正常' : '偏高', unit: '',
        status: (metrics.step_time_cv || 99) < 15 ? 'good' : 'danger' },
      { label: '足廓清', value: (metrics.foot_clearance_cm || 0).toFixed(1), unit: 'cm',
        status: (metrics.foot_clearance_cm || 0) > 3 ? 'good' : 'warn' },
    ];
    this.setData({
      grsScore: Math.round(metrics.gait_rehab_score || 0),
      metricsList: list
    });
    this._drawGauge(metrics.gait_rehab_score || 0);
  },

  _drawGauge(score) {
    const query = wx.createSelectorQuery();
    query.select('#gaugeCanvas').fields({ node: true, size: true }).exec((res) => {
      if (!res[0] || !res[0].node) return;
      const canvas = res[0].node;
      const ctx = canvas.getContext('2d');
      const dpr = wx.getSystemInfoSync().pixelRatio;
      canvas.width = 320 * dpr;
      canvas.height = 200 * dpr;
      ctx.scale(dpr, dpr);

      const cx = 160, cy = 135, r = 95;
      const color = score < 50 ? '#e74c3c' : score < 70 ? '#f39c12' : '#27ae60';

      // 背景弧
      ctx.beginPath();
      ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI);
      ctx.strokeStyle = '#e8e8e8';
      ctx.lineWidth = 14;
      ctx.lineCap = 'round';
      ctx.stroke();

      // 前景弧
      const endAngle = Math.PI + (Math.min(score, 100) / 100) * Math.PI;
      ctx.beginPath();
      ctx.arc(cx, cy, r, Math.PI, endAngle);
      ctx.strokeStyle = color;
      ctx.stroke();

      // 分数
      ctx.fillStyle = color;
      ctx.font = 'bold 36px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(Math.round(score)), cx, cy - 10);
    });
  },

  _drawEmotionChart(stats) {
    const query = wx.createSelectorQuery();
    query.select('#emotionCanvas').fields({ node: true, size: true }).exec((res) => {
      if (!res[0] || !res[0].node || !stats) return;
      const canvas = res[0].node;
      const ctx = canvas.getContext('2d');
      const dpr = wx.getSystemInfoSync().pixelRatio;
      const w = res[0].width, h = 200 * dpr / dpr || 200;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.scale(dpr, dpr);

      // 聚合各情绪总数
      const totals = {};
      Object.values(stats).forEach(day => {
        Object.entries(day).forEach(([label, cnt]) => {
          totals[label] = (totals[label] || 0) + cnt;
        });
      });
      const labels = Object.keys(totals);
      if (labels.length === 0) return;
      const maxCnt = Math.max(...Object.values(totals), 1);

      const barW = (w - 40) / labels.length - 8;
      const colors = ['#e74c3c','#8e44ad','#3498db','#f1c40f','#2ecc71','#95a5a6','#e67e22'];

      ctx.clearRect(0, 0, w, h);
      labels.forEach((label, i) => {
        const x = 20 + i * (barW + 8);
        const barH = (totals[label] / maxCnt) * (h - 50);
        const y = h - 20 - barH;

        ctx.fillStyle = colors[i % colors.length];
        ctx.fillRect(x, y, barW, barH);

        ctx.fillStyle = '#666';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(label, x + barW / 2, h - 4);
      });
    });
  },

  // ===== 跌倒告警（由 app.js 事件总线触发）=====

  onFallAlert(data) {
    wx.showModal({
      title: '跌倒告警',
      content: '检测到跌倒事件！位置: (' +
        ((data.location && data.location[0]) || 0).toFixed(1) + ', ' +
        ((data.location && data.location[1]) || 0).toFixed(1) + ')',
      confirmText: '知道了',
      showCancel: false
    });
  },

  // ===== 导航 =====

  goToCamera() { wx.navigateTo({ url: '/pages/camera/camera' }); },
  goToTrend() { wx.switchTab({ url: '/pages/trend/trend' }); },
  goToMap() { wx.switchTab({ url: '/pages/map/map' }); },
  goToReport() { wx.navigateTo({ url: '/pages/report/report' }); },
  goToChat() { wx.navigateTo({ url: '/pages/chat/chat' }); },
  goToProfile() { wx.switchTab({ url: '/pages/profile/profile' }); },

  // ===== 按钮 =====

  unlockPatient() {
    wx.showModal({
      title: '取消锁定',
      content: '将清除后端锁定状态和人脸数据，确定取消？',
      confirmText: '确定取消',
      confirmColor: '#e74c3c',
      success: async (res) => {
        if (res.confirm) {
          try {
            await api.unlockPatient();
            app._clearLockState();
            this._stopPolling();
            this.setData({
              locked: false,
              patientName: '',
              grsScore: 0,
              metricsList: []
            });
            wx.showToast({ title: '已取消锁定', icon: 'success' });
          } catch (e) {
            wx.showToast({ title: '请求失败，请重试', icon: 'none' });
          }
        }
      }
    });
  },

  quickReport() {
    wx.showLoading({ title: '生成中...' });
    api.generateReport(30).then(res => {
      wx.hideLoading();
      wx.navigateTo({ url: '/pages/report/report?text=' + encodeURIComponent(res.text || '') });
    }).catch(err => {
      wx.hideLoading();
      wx.showToast({ title: err.message || '生成失败', icon: 'none' });
    });
  },

  emergency() {
    wx.showModal({
      title: '紧急求助',
      content: '是否拨打紧急电话？',
      confirmText: '拨打',
      success(res) {
        if (res.confirm) wx.makePhoneCall({ phoneNumber: '120' });
      }
    });
  },

  onShareAppMessage() {
    return { title: '康复监测 - 实时步态分析' };
  },

  _stopPolling() {
    this._polling = false;
    clearInterval(this.pollTimer);
    this.pollTimer = null;
  },

  onUnload() {
    this._stopPolling();
    clearInterval(this._statusTimer);
  }
});
