const app = getApp();

Page({
  data: {
    patientId: '',
    locked: false,
    notificationsEnabled: true,
    apiBaseUrl: ''
  },

  onLoad() {
    this.setData({
      patientId: app.globalData.patientId || '未锁定',
      locked: app.globalData.patientLocked,
      apiBaseUrl: wx.getStorageSync('apiBaseUrl') || '192.168.1.100:5000'
    });
  },

  onShow() {
    this.setData({
      patientId: app.globalData.patientId || '未锁定',
      locked: app.globalData.patientLocked
    });
  },

  toggleNotification(e) {
    this.setData({ notificationsEnabled: !!e.detail.value });
  },

  changeApiBase() {
    const current = this.data.apiBaseUrl;
    wx.showModal({
      title: '修改服务器地址',
      editable: true,
      placeholderText: '192.168.1.100:5000',
      content: current,
      success: (res) => {
        if (res.confirm && res.content) {
          wx.setStorageSync('apiBaseUrl', res.content.trim());
          this.setData({ apiBaseUrl: res.content.trim() });
        }
      }
    });
  },

  switchPatient() {
    wx.navigateTo({ url: '/pages/camera/camera' });
  }
});
