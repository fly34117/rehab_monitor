/**
 * WebSocket 连接管理
 * - 自动重连 (断线后每5秒重试，最多10次)
 * - 心跳保活 (30秒 ping/pong)
 * - 多消息类型分发 (metrics / fall_alert)
 */

class MetricsSocket {
  constructor() {
    this.socket = null;
    this.url = '';
    this.callbacks = [];
    this._statusCallbacks = [];
    this.reconnectTimer = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
    this.heartbeatTimer = null;
    this.connected = false;
  }

  connect(url) {
    this.url = url;
    this._doConnect();
  }

  _doConnect() {
    // 先清理旧连接，避免 closeSocket task not found
    if (this.socket) {
      try { this.socket.close({ code: 1000, reason: 'reconnect' }); } catch (e) {}
      this.socket = null;
    }

    this.socket = wx.connectSocket({
      url: this.url,
      fail: () => this._onClose()
    });

    this.socket.onOpen(() => {
      this.connected = true;
      this.reconnectAttempts = 0;
      this._startHeartbeat();
      this._notifyStatus(true);
    });

    this.socket.onMessage((res) => {
      try {
        const msg = JSON.parse(res.data);
        if (msg.type === 'ping') {
          if (this.socket) this.socket.send({ data: 'pong' });
          return;
        }
        // 分发给所有订阅者
        this.callbacks.forEach(cb => {
          try { cb(msg); } catch (e) {}
        });
      } catch (e) {}
    });

    this.socket.onClose(() => this._onClose());
    this.socket.onError(() => this._onClose());
  }

  /** 订阅消息回调 */
  subscribe(callback) {
    if (typeof callback === 'function') {
      this.callbacks.push(callback);
    }
  }

  /** 取消订阅 */
  unsubscribe(callback) {
    this.callbacks = this.callbacks.filter(cb => cb !== callback);
  }

  /** 订阅连接状态变化（即时推送，不走轮询） */
  onStatusChange(callback) {
    if (typeof callback === 'function') {
      this._statusCallbacks.push(callback);
    }
  }

  _notifyStatus(connected) {
    this._statusCallbacks.forEach(cb => {
      try { cb(connected); } catch (e) {}
    });
  }

  _startHeartbeat() {
    this._stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (this.socket && this.connected) {
        this.socket.send({ data: 'ping' });
      }
    }, 30000);
  }

  _stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  _onClose() {
    this.connected = false;
    this._stopHeartbeat();
    this.socket = null;
    this._notifyStatus(false);

    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      this.reconnectTimer = setTimeout(() => {
        this._doConnect();
      }, 5000);
    }
  }

  disconnect() {
    this.maxReconnectAttempts = 0;  // 阻止重连
    this._stopHeartbeat();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.socket) {
      try { this.socket.close({ code: 1000, reason: 'disconnect' }); } catch (e) {}
      this.socket = null;
    }
    this.connected = false;
    this.callbacks = [];
    this._statusCallbacks = [];
    this._notifyStatus(false);
  }
}

module.exports = { MetricsSocket };
