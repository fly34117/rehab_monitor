const app = getApp();

Page({
  data: {
    patientId: '',
    locked: false,
    notificationsEnabled: true,
    apiBaseUrl: '',
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
  },

  scanQRCode() {
    wx.scanCode({
      scanType: ['qrCode'],
      success: (res) => {
        try {
          const data = JSON.parse(res.result);
          if (data.ip && data.port) {
            const addr = `${data.ip}:${data.port}`;
            wx.setStorageSync('apiBaseUrl', addr);
            this.setData({ apiBaseUrl: addr });
            wx.showToast({ title: `已连接 ${data.ip}`, icon: 'success' });
          } else {
            wx.showToast({ title: '无效的二维码', icon: 'error' });
          }
        } catch (e) {
          wx.showToast({ title: '二维码格式错误', icon: 'error' });
        }
      },
      fail: () => {
        wx.showToast({ title: '扫码取消', icon: 'none' });
      }
    });
  },
});
