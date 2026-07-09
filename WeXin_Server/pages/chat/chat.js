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

    // 用户气泡 + AI 占位气泡
    const messages = [
      ...this.data.messages,
      { role: 'user', content: msg },
      { role: 'assistant', content: '▊', streaming: true },
    ];
    this.setData({ messages });
    this._scrollToBottom();

    // 流式请求
    api.sendChatStream(
      msg,
      // onChunk: 逐字更新助手气泡
      (text) => {
        const msgs = [...this.data.messages];
        // 更新最后一个助手气泡
        const last = msgs[msgs.length - 1];
        if (last && last.role === 'assistant') {
          last.content = text + '▊';
          last.streaming = true;
        }
        this.setData({ messages: msgs });
      },
      // onDone: 完成
      (finalText) => {
        const msgs = [...this.data.messages];
        const last = msgs[msgs.length - 1];
        if (last && last.role === 'assistant') {
          last.content = finalText || last.content.replace('▊', '');
          last.streaming = false;
        }
        this.setData({ messages: msgs, sending: false });
        this._scrollToBottom();
      },
      // onError
      (errMsg) => {
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
