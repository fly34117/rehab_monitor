const api = require('../../utils/api');
const app = getApp();

Page({
  data: {
    messages: [],
    loading: false,
    locked: false,
    inputText: '',
    sending: false,
  },

  onShow() {
    this.setData({ locked: app.globalData.patientLocked });
    if (app.globalData.patientLocked) {
      this.loadHistory();
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
    } catch (e) {
      this.setData({ loading: false });
      wx.showToast({ title: '加载对话失败', icon: 'none' });
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

    // 先乐观显示用户消息
    const messages = [...this.data.messages, { role: 'user', content: msg }];
    this.setData({ messages });
    this._scrollToBottom();

    try {
      const result = await api.sendChat(msg);
      // 追加助手回复
      messages.push({ role: 'assistant', content: result.reply || '' });
      this.setData({ messages, sending: false });
      this._scrollToBottom();
    } catch (e) {
      wx.showToast({ title: '发送失败', icon: 'none' });
      this.setData({ sending: false });
    }
  },

  _scrollToBottom() {
    wx.createSelectorQuery()
      .select('.chat-list')
      .boundingClientRect()
      .exec(() => {
        wx.pageScrollTo({ scrollTop: 99999, duration: 200 });
      });
  },
});
