const api = require('../../utils/api');
const app = getApp();

Page({
  data: {
    messages: [],
    loading: false,
    locked: false,
    inputText: '',
    sending: false,
    scrollIntoView: '',
  },

  onShow() {
    this.setData({ locked: app.globalData.patientLocked });
    if (app.globalData.patientLocked) {
      this.loadHistory();
    } else {
      this._stopPolling();
    }
  },

  onHide() {
    this._stopPolling();
  },

  onUnload() {
    this._stopPolling();
  },

  _startPolling() {
    this._stopPolling();
    this._pollTimer = setInterval(() => {
      this._pollHistory();
    }, 3000);
  },

  _stopPolling() {
    if (this._pollTimer) {
      clearInterval(this._pollTimer);
      this._pollTimer = null;
    }
  },

  async _pollHistory() {
    if (this.data.sending) return;
    try {
      const result = await api.getChatHistory();
      const msgs = result.messages || [];
      // 服务端有新消息或内容不同时更新（>= 保证流式异常结束时占位符不被服务端未完成状态覆盖）
      if (msgs.length >= this.data.messages.length) {
        this.setData({ messages: msgs });
        this._scrollToBottom();
      }
    } catch (e) {
      // 静默失败，不打扰用户
    }
  },

  onPullDownRefresh() {
    this.loadHistory().then(() => wx.stopPullDownRefresh());
  },

  async loadHistory() {
    this.setData({ loading: true });
    try {
      const result = await api.getChatHistory();
      this.setData({
        messages: result.messages || [],
        loading: false,
      });
      this._scrollToBottom();
      this._startPolling();
    } catch (e) {
      this.setData({ loading: false });
    }
  },

  clearChat() {
    wx.showModal({
      title: '清除对话',
      content: '确定要清除所有对话记录吗？',
      success: async (res) => {
        if (res.confirm) {
          try {
            await api.clearChat();
            this.setData({ messages: [] });
            wx.showToast({ title: '已清除', icon: 'success' });
          } catch (e) {
            wx.showToast({ title: '清除失败', icon: 'none' });
          }
        }
      }
    });
  },

  onInput(e) {
    this.setData({ inputText: e.detail.value });
  },

  async sendMessage() {
    const msg = this.data.inputText.trim();
    if (!msg || this.data.sending) return;

    this.setData({ inputText: '', sending: true });

    // 用户气泡 + AI 占位气泡
    const messages = [
      ...this.data.messages,
      { role: 'user', content: msg },
      { role: 'assistant', content: '▊', streaming: true },
    ];
    this.setData({ messages });
    this._scrollToBottom();

    // 节流：最多每 80ms 更新一次 setData，避免竞态 + 提高性能
    let _lastUpdate = 0;
    let _pendingText = '▊';
    let _timer = null;
    const _flushDisplay = () => {
      const msgs = [...this.data.messages];
      const last = msgs[msgs.length - 1];
      if (last && last.role === 'assistant') {
        last.content = _pendingText;
        last.streaming = !!_pendingText.includes('▊');
      }
      this.setData({ messages: msgs });
    };

    // 流式请求
    api.sendChatStream(
      msg,
      // onChunk: 收到增量 → 本地累积 + 节流刷新
      (text) => {
        _pendingText = text + '▊';
        const now = Date.now();
        if (now - _lastUpdate >= 80) {
          _lastUpdate = now;
          _flushDisplay();
        } else if (!_timer) {
          _timer = setTimeout(() => {
            _lastUpdate = Date.now();
            _flushDisplay();
            _timer = null;
          }, 80);
        }
      },
      // onDone: 完成
      (finalText) => {
        if (_timer) { clearTimeout(_timer); _timer = null; }
        if (finalText) {
          const msgs = [...this.data.messages];
          const last = msgs[msgs.length - 1];
          if (last && last.role === 'assistant') {
            last.content = finalText;
            last.streaming = false;
          }
          this.setData({ messages: msgs });
          this._scrollToBottom();
        }
        // finalText 为空：流式异常结束，保留占位符，轮询会恢复
        this.setData({ sending: false });
      },
      // onError
      (errMsg) => {
        if (_timer) { clearTimeout(_timer); _timer = null; }
        const msgs = [...this.data.messages];
        const last = msgs[msgs.length - 1];
        if (last && last.role === 'assistant') {
          last.content = '抱歉，请求失败：' + (errMsg || '未知错误');
          last.streaming = false;
        }
        this.setData({ messages: msgs, sending: false });
      }
    );
  },

  async generateReport() {
    if (this.data.sending) return;
    this.setData({ sending: true });

    const messages = [...this.data.messages, { role: 'user', content: '📋 请求生成康复分析报告' }];
    this.setData({ messages });
    this._scrollToBottom();

    try {
      const result = await api.generateReport(30);
      const reply = result.text || '暂无报告内容';
      messages.push({ role: 'assistant', content: reply });
      this.setData({ messages, sending: false });
      this._scrollToBottom();
    } catch (e) {
      wx.showToast({ title: e.message || '生成失败', icon: 'none', duration: 1500 });
      this.setData({ sending: false });
    }
  },

  _scrollToBottom() {
    this.setData({ scrollIntoView: '' });
    setTimeout(() => {
      this.setData({ scrollIntoView: 'chat-bottom' });
    }, 50);
  },
});
