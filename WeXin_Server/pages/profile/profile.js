const app = getApp();
const { scanLan } = require('../../utils/discovery');

Page({
  data: {
    patientId: '',
    locked: false,
    notificationsEnabled: true,
    apiBaseUrl: '',
    scanning: false,
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

  async scanLan() {
    this.setData({ scanning: true });
    try {
      const servers = await scanLan(2500);
      this.setData({ scanning: false });

      if (servers.length === 0) {
        wx.showToast({ title: '未发现设备，请确认同一WiFi', icon: 'none' });
        return;
      }

      if (servers.length === 1) {
        // 只找到一个，直接连接
        const s = servers[0];
        const addr = `${s.ip}:${s.port}`;
        wx.setStorageSync('apiBaseUrl', addr);
        this.setData({ apiBaseUrl: addr });
        wx.showToast({ title: `已连接 ${s.hostname || s.ip}`, icon: 'success' });
      } else {
        // 多个服务器，让用户选
        const items = servers.map(s => `${s.hostname || s.ip} (${s.ip})`);
        wx.showActionSheet({
          itemList: items,
          success: (res) => {
            const s = servers[res.tapIndex];
            const addr = `${s.ip}:${s.port}`;
            wx.setStorageSync('apiBaseUrl', addr);
            this.setData({ apiBaseUrl: addr });
            wx.showToast({ title: `已连接 ${s.hostname || s.ip}`, icon: 'success' });
          }
        });
      }
    } catch (e) {
      this.setData({ scanning: false });
      wx.showToast({ title: '扫描失败，请重试', icon: 'none' });
    }
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
