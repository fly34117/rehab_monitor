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
      if (msgs.length !== this.data.messages.length) {
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

    const messages = [...this.data.messages, { role: 'user', content: msg }];
    this.setData({ messages });
    this._scrollToBottom();

    try {
      const result = await api.sendChat(msg);
      messages.push({ role: 'assistant', content: result.reply || '' });
      this.setData({ messages, sending: false });
      this._scrollToBottom();
    } catch (e) {
      wx.showToast({ title: e.message || '发送失败', icon: 'none', duration: 1500 });
      this.setData({ sending: false });
    }
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
