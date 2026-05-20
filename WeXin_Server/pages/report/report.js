const api = require('../../utils/api');
const app = getApp();

Page({
  data: {
    seconds: 30,
    reportText: '',
    reportSummary: '',
    loading: false,
    locked: false
  },

  onLoad() {
    this.setData({ locked: app.globalData.patientLocked });
  },

  onShow() {
    this.setData({ locked: app.globalData.patientLocked });
  },

  changePeriod(e) {
    this.setData({ seconds: parseInt(e.currentTarget.dataset.seconds) });
  },

  async generateReport() {
    this.setData({ loading: true });
    try {
      const result = await api.generateReport(this.data.seconds);
      this.setData({
        reportText: result.text || '暂无报告内容',
        reportSummary: result.summary || '',
        loading: false
      });
    } catch (e) {
      this.setData({ loading: false });
      wx.showToast({ title: e.message || '报告生成失败', icon: 'error' });
    }
  },

  onShareAppMessage() {
    return {
      title: '康复报告 - ' + (this.data.reportSummary || '步态分析'),
      path: '/pages/report/report'
    };
  }
});
