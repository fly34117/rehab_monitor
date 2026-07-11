/**
 * API 请求封装 — 康复监测系统
 * 所有请求携带 X-Session-Id 头，用于服务端限流区分设备
 */

// 根据 PC 实际局域网 IP 修改
// 从本地存储读取服务器地址，默认使用当前 PC 局域网 IP
const getApiBase = () => {
  const addr = wx.getStorageSync('apiBaseUrl') || '192.168.46.21:5000';
  return 'http://' + addr + '/api/v1';
};
const SESSION_ID = 'wx_' + Date.now();

function request(method, path, data = null) {
  return new Promise((resolve, reject) => {
    wx.request({
      url: getApiBase() + path,
      method: method,
      data: data,
      header: {
        'Content-Type': 'application/json',
        'X-Session-Id': SESSION_ID
      },
      success(res) {
        if (res.statusCode === 200 && res.data.code === 0) {
          resolve(res.data.data);
        } else if (res.statusCode === 429) {
          reject({ code: -1, message: '请求过于频繁，请稍后再试' });
        } else {
          reject(res.data || { code: -1, message: '请求失败' });
        }
      },
      fail(err) {
        reject({ code: -1, message: '网络请求失败', error: err });
      }
    });
  });
}

/** 获取实时步态指标 (23项) */
function getRealtimeMetrics() {
  return request('GET', '/metrics/realtime');
}

/** 获取历史趋势 (日均值) */
function getHistoryTrend(days = 7) {
  return request('GET', '/metrics/history?days=' + days);
}

/** 拍照锁定患者 (自动判断录入/匹配) */
function lockPatient(imageBase64) {
  return request('POST', '/patient/lock', { image: imageBase64 });
}

/** 取消锁定 */
function unlockPatient() {
  return request('POST', '/patient/unlock');
}

/** 生成康复报告 (快速模式) */
function generateReport(seconds = 30) {
  return request('POST', '/report/generate?seconds=' + seconds);
}

/** 生成专家知识库报告 (权威论文 + DeepSeek) */
function generateExpertReport(seconds = 60, trendDays = 0) {
  let path = '/report/expert?seconds=' + seconds;
  if (trendDays > 0) path += '&trend_days=' + trendDays;
  return request('POST', path);
}

/** 获取心情统计 (按天+情绪聚合) */
function getEmotionStats(days = 7) {
  return request('GET', '/emotion/stats?days=' + days);
}

/** 获取位置轨迹 */
function getTrajectory(limit = 100) {
  return request('GET', '/trajectory?limit=' + limit);
}

/** 获取本地 LLM 对话历史 */
function getChatHistory() {
  return request('GET', '/chat/history');
}

/** 清除本地 LLM 对话历史 */
function clearChat() {
  return request('POST', '/chat/clear');
}

/** 发送消息到本地 LLM */
function sendChat(message) {
  return request('POST', '/chat/send', { message: message });
}

/** 流式发送消息 — token-by-token 回调 */
function sendChatStream(message, onChunk, onDone, onError) {
  let accumulated = '';
  let done = false;

  const requestTask = wx.request({
    url: getApiBase() + '/chat/stream',
    method: 'POST',
    data: { message: message },
    header: {
      'Content-Type': 'application/json',
      'X-Session-Id': SESSION_ID
    },
    enableChunked: true,
    success(res) {
      if (done) return;
      done = true;
      // 兜底: 如果还有剩余文本没回调过，补推一次
      if (accumulated && onChunk) {
        onChunk(accumulated);
      }
      // 最终文本 = 去掉 ERROR 行后的内容
      const cleaned = accumulated.replace(/\n?ERROR:.*$/, '');
      if (onDone) onDone(cleaned);
    },
    fail(err) {
      if (done) return;
      done = true;
      if (onError) onError(err.errMsg || '网络请求失败');
    }
  });

  if (requestTask && requestTask.onChunkReceived) {
    requestTask.onChunkReceived((chunk) => {
      if (done) return;
      try {
        // 正确的 UTF-8 解码（避免 apply 参数上限 + 中文断裂）
        const arr = new Uint8Array(chunk.data);
        let text = '';
        // 逐段拼接避免 call stack overflow
        for (let i = 0; i < arr.length; i += 4096) {
          text += String.fromCharCode.apply(null, Array.from(arr.slice(i, i + 4096)));
        }
        // 用 TextDecoder 备选（基础库 3.0+ 支持）
        if (typeof TextDecoder !== 'undefined') {
          try {
            text = new TextDecoder('utf-8').decode(arr);
          } catch (e) { /* 回退到上面的方案 */ }
        }
        if (!text) return;

        // 检查是否包含错误标记
        if (text.includes('ERROR:')) {
          done = true;
          const errMatch = text.match(/ERROR:(.*)/);
          if (errMatch && errMatch[1] && onError) {
            onError(errMatch[1]);
          }
          return;
        }

        accumulated += text;
        if (onChunk) onChunk(accumulated);
      } catch (e) {}
    });
  }

  return requestTask;
}

module.exports = {
  getRealtimeMetrics,
  getHistoryTrend,
  lockPatient,
  unlockPatient,
  generateReport,
  generateExpertReport,
  getEmotionStats,
  getTrajectory,
  getChatHistory,
  clearChat,
  sendChat,
  sendChatStream,
};
