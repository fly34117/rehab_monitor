const api = require('../../utils/api');
const app = getApp();

Page({
  data: {
    messages: [],
    loading: false,
    locked: false,
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
});
