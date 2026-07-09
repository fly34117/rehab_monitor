const api = require('../../utils/api');
const app = getApp();

Page({
  data: {
    loading: false,
    cameraPosition: 'back'  // 'front' | 'back'
  },

  flipCamera() {
    this.setData({
      cameraPosition: this.data.cameraPosition === 'front' ? 'back' : 'front'
    });
  },

  takePhoto() {
    if (this.data.loading) return;
    const ctx = wx.createCameraContext();
    ctx.takePhoto({
      quality: 'low',
      success: (res) => this._compressAndLock(res.tempImagePath),
      fail: () => {
        wx.showToast({ title: '拍照失败，请检查相机权限', icon: 'none' });
      }
    });
  },

  chooseFromAlbum() {
    wx.chooseImage({
      count: 1,
      sizeType: ['compressed'],
      sourceType: ['album'],
      success: (res) => this._compressAndLock(res.tempFilePaths[0])
    });
  },

  _compressAndLock(filePath) {
    if (!filePath) {
      wx.showToast({ title: '图片获取失败', icon: 'none' });
      return;
    }
    wx.compressImage({
      src: filePath,
      quality: 60,
      success: (compressRes) => {
        const readPath = compressRes.tempFilePath || filePath;
        try {
          const fs = wx.getFileSystemManager();
          const base64 = fs.readFileSync(readPath, 'base64');
          this._lockPatient('data:image/jpeg;base64,' + base64);
        } catch (e) {
          wx.showToast({ title: '图片读取失败', icon: 'none' });
        }
      },
      fail: () => {
        // DevTools 不支持压缩，直接用原图
        try {
          const fs = wx.getFileSystemManager();
          const base64 = fs.readFileSync(filePath, 'base64');
          this._lockPatient('data:image/jpeg;base64,' + base64);
        } catch (e) {
          wx.showToast({ title: '图片读取失败', icon: 'none' });
        }
      }
    });
  },

  async _lockPatient(imageBase64) {
    this.setData({ loading: true });
    wx.showLoading({ title: '识别中...' });

    try {
      const result = await api.lockPatient(imageBase64);
      wx.hideLoading();
      this.setData({ loading: false });

      app.setPatientLocked(result.patientId, result.patientId);

      wx.showToast({
        title: result.message || '锁定成功',
        icon: 'success'
      });

      setTimeout(() => wx.switchTab({ url: '/pages/index/index' }), 1500);
    } catch (e) {
      wx.hideLoading();
      this.setData({ loading: false });
      wx.showToast({
        title: e.message || '锁定失败',
        icon: 'error',
        duration: 2000
      });
    }
  },

  onCameraError() {
    wx.showToast({ title: '摄像头不可用', icon: 'none' });
  }
});
