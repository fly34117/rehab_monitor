/**
 * 局域网 mDNS 服务发现
 * 
 * PC 服务端通过 zeroconf 注册 _rehab._tcp 服务，
 * 小程序用 wx.startLocalServiceDiscovery 自动发现。
 */

const SCAN_TIMEOUT = 3000;  // 扫描超时（ms）

/**
 * 扫描局域网内的康复监测服务器
 * @param {number} timeout 扫描超时（ms），默认 3000
 * @returns {Promise<Array<{ip, port, hostname}>>}
 */
function scanLan(timeout = SCAN_TIMEOUT) {
  return new Promise((resolve) => {
    const servers = [];
    const seen = new Set();
    let service = null;
    let timer = null;

    const finish = () => {
      if (timer) clearTimeout(timer);
      if (service) {
        try { service.stop(); } catch (e) {}
      }
      resolve(servers);
    };

    timer = setTimeout(finish, timeout);

    try {
      service = wx.startLocalServiceDiscovery({
        serviceType: '_rehab._tcp.local.',

        success: () => {
          service.onFound((res) => {
            try {
              const ip = res.ip || '';
              const port = (res.service && res.service.port) || 5000;
              const attrs = res.service && res.service.attributes || {};
              const hostname = attrs.hostname || res.serviceName || '';

              if (!ip) return;
              const key = `${ip}:${port}`;
              if (seen.has(key)) return;
              seen.add(key);

              servers.push({ ip, port, hostname });
            } catch (e) {}
          });

          // 发现完成（部分 Android 可能不触发，超时兜底）
          service.onServiceResolveComplete && service.onServiceResolveComplete(() => {});
        },

        fail: (err) => {
          // mDNS 不可用（如 iOS 限制），直接结束
          finish();
        },
      });
    } catch (e) {
      finish();
    }
  });
}

module.exports = { scanLan };
