const api = require('../../utils/api');
const app = getApp();

Page({
  data: {
    // 模式: 'quick' | 'expert'
    mode: 'quick',
    seconds: 30,
    trendDays: 7,
    reportText: '',
    reportSummary: '',
    reportJson: null,
    citedCount: 0,
    loading: false,
    locked: false
  },

  onLoad(options) {
    this.setData({ locked: app.globalData.patientLocked });
    // 处理从首页快速报告传来的 text 参数
    if (options && options.text) {
      const text = decodeURIComponent(options.text);
      this.setData({
        mode: 'quick',
        reportText: text,
        reportSummary: text.substring(0, 100)
      });
    }
  },

  onShow() {
    this.setData({ locked: app.globalData.patientLocked });
  },

  switchMode(e) {
    const mode = e.currentTarget.dataset.mode;
    this.setData({
      mode: mode,
      reportText: '',
      reportSummary: '',
      reportJson: null,
      citedCount: 0
    });
  },

  changePeriod(e) {
    this.setData({ seconds: parseInt(e.currentTarget.dataset.seconds) });
  },

  changeTrendDays(e) {
    this.setData({ trendDays: parseInt(e.currentTarget.dataset.days) });
  },

  async generateReport() {
    this.setData({ loading: true });
    try {
      if (this.data.mode === 'expert') {
        // 专家知识库分析
        const result = await api.generateExpertReport(
          this.data.seconds,
          this.data.trendDays
        );
        this.setData({
          reportText: result.text || '暂无报告内容',
          reportSummary: result.summary || '',
          reportJson: result.report_json || null,
          citedCount: result.cited_count || 0,
          loading: false
        });
      } else {
        // 快速分析
        const result = await api.generateReport(this.data.seconds);
        this.setData({
          reportText: result.text || '暂无报告内容',
          reportSummary: result.summary || '',
          reportJson: result.summary || null,
          citedCount: 0,
          loading: false
        });
      }
    } catch (e) {
      this.setData({ loading: false });
      wx.showToast({ title: e.message || '报告生成失败', icon: 'error' });
    }
  },

  onShareAppMessage() {
    const title = this.data.mode === 'expert'
      ? '专家步态分析报告'
      : '康复报告 - ' + (this.data.reportSummary || '步态分析');
    return {
      title: title,
      path: '/pages/report/report'
    };
  }
});
