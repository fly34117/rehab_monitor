const { MetricsSocket } = require('./utils/websocket');

App({
  globalData: {
    patientLocked: false,
    patientId: null,
    patientName: '',
    lastMetrics: null,
    lastEmotionStats: null,
    offlineMode: false,
    pendingRequests: [],
    wsConnected: false
  },

  socket: null,

  onLaunch() {
    // 先信任本地存储（快速显示），再异步验证后端状态
    const locked = wx.getStorageSync('patientLocked');
    if (locked) {
      this.globalData.patientLocked = true;
      this.globalData.patientId = wx.getStorageSync('patientId') || '';
      this.globalData.patientName = wx.getStorageSync('patientName') || '';
    }

    // 网络状态监听
    wx.onNetworkStatusChange((res) => {
      this.globalData.offlineMode = !res.isConnected;
      if (res.isConnected) {
        this._flushPendingRequests();
        if (this.globalData.patientLocked && !this.globalData.wsConnected) {
          this._initWebSocket();
        }
      }
    });

    // 如果本地显示已锁定，异步验证后端状态
    if (this.globalData.patientLocked) {
      this._verifyLockState();
    }
  },

  /** 向后端健康检查接口验证锁定状态是否仍然有效 */
  _verifyLockState() {
    const apiBase = this._getApiBase();
    wx.request({
      url: apiBase + '/api/v1/health',
      method: 'GET',
      success: (res) => {
        if (res.statusCode === 200 && res.data.code === 0) {
          const backendLocked = res.data.data.target_locked;
          if (backendLocked && backendLocked.locked) {
            // 后端确认锁定 → 更新本地状态 + 启动 WebSocket
            this.globalData.patientLocked = true;
            this.globalData.patientId = backendLocked.name;
            this.globalData.patientName = backendLocked.name;
            wx.setStorageSync('patientLocked', true);
            wx.setStorageSync('patientId', backendLocked.name);
            wx.setStorageSync('patientName', backendLocked.name);
            this._initWebSocket();
          } else {
            // 后端未锁定 → 清除本地残留状态
            this._clearLockState();
          }
        }
      },
      fail: () => {
        // 网络不通 → 保留本地状态，启动 WebSocket 尝试
        this._initWebSocket();
      }
    });
  },

  /** 清除本地锁定状态 */
  _clearLockState() {
    this.globalData.patientLocked = false;
    this.globalData.patientId = null;
    this.globalData.patientName = '';
    wx.removeStorageSync('patientLocked');
    wx.removeStorageSync('patientId');
    wx.removeStorageSync('patientName');
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  },

  _getApiBase() {
    const addr = wx.getStorageSync('apiBaseUrl') || '192.168.249.179:5000';
    return 'http://' + addr;
  },

  _initWebSocket() {
    if (this.socket) {
      this.socket.disconnect();
    }
    this.socket = new MetricsSocket();

    this.socket.subscribe((msg) => {
      if (msg.type === 'metrics') {
        this.globalData.lastMetrics = msg.data;
      }
      if (msg.type === 'fall_alert') {
        this._emitFallAlert(msg.data);
      }
    });

    const wsUrl = this._getWsUrl();
    this.socket.connect(wsUrl);

    this._wsCheckTimer = setInterval(() => {
      this.globalData.wsConnected = this.socket && this.socket.connected;
    }, 5000);
  },

  _getWsUrl() {
    const addr = wx.getStorageSync('apiBaseUrl') || '192.168.249.179:5000';
    // REST API 在 5000，WebSocket 在 5001（websockets 独立端口）
    const wsAddr = addr.replace(/:5000$/, ':5001');
    return 'ws://' + wsAddr;
  },

  _flushPendingRequests() {
    const list = [...this.globalData.pendingRequests];
    this.globalData.pendingRequests = [];
    list.forEach(req => {
      wx.request(req);
    });
  },

  _emitFallAlert(data) {
    const pages = getCurrentPages();
    pages.forEach(page => {
      if (typeof page.onFallAlert === 'function') {
        try { page.onFallAlert(data); } catch (e) {}
      }
    });
  },

  setPatientLocked(patientId, patientName) {
    // 先保存目标信息，等待后端确认锁定后再启用完整功能
    this.globalData.patientLocked = true;
    this.globalData.patientId = patientId;
    this.globalData.patientName = patientName;
    wx.setStorageSync('patientLocked', true);
    wx.setStorageSync('patientId', patientId);
    wx.setStorageSync('patientName', patientName);
    this._initWebSocket();

    // 轮询后端确认锁定完成（搜索 → 实际锁定，最多等 15 秒）
    let attempts = 0;
    const maxAttempts = 15;
    const checkLocked = () => {
      if (attempts >= maxAttempts) return;
      attempts++;
      wx.request({
        url: this._getApiBase() + '/api/v1/health',
        method: 'GET',
        success: (res) => {
          if (res.statusCode === 200 && res.data.code === 0) {
            const backend = res.data.data.target_locked;
            if (backend && backend.locked) {
              // 后端确认锁定 → 更新状态并通知所有页面刷新
              this.globalData._backendLockConfirmed = true;
              this._notifyPages('backendLocked');
              return;
            }
          }
          // 尚未锁定，1 秒后重试
          setTimeout(checkLocked, 1000);
        },
        fail: () => {
          setTimeout(checkLocked, 1000);
        }
      });
    };
    setTimeout(checkLocked, 1500);
  },

  /** 通知所有活跃页面 */
  _notifyPages(event) {
    const pages = getCurrentPages();
    pages.forEach(page => {
      if (typeof page.onAppEvent === 'function') {
        try { page.onAppEvent(event); } catch (e) {}
      }
    });
  },
});
