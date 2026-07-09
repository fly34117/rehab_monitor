/**
 * 局域网 HTTP 子网扫描发现
 *
 * 直接向常见 IP 的 /api/v1/health 发请求验证。
 * 先扫最可能的 IP（路由器 DHCP 通常从 .100 开始），
 * 找到后立即停止。
 */

const HEALTH_PATH = '/api/v1/health';

// 优先扫描：最可能分配到的地址
// 大多数路由器 DHCP 池从 .100 开始
const PRIORITY_TARGETS = [
  // 192.168.1.x（TP-Link, ASUS, 小米）
  '192.168.1.100', '192.168.1.101', '192.168.1.102', '192.168.1.103',
  '192.168.1.104', '192.168.1.105', '192.168.1.106', '192.168.1.107',
  '192.168.1.108', '192.168.1.109', '192.168.1.110',
  // 192.168.0.x（D-Link, NETGEAR）
  '192.168.0.100', '192.168.0.101', '192.168.0.102', '192.168.0.103',
  '192.168.0.104', '192.168.0.105', '192.168.0.106', '192.168.0.107',
  // 其他常见
  '192.168.31.100', '192.168.2.100', '192.168.50.100',
  '192.168.1.1', '192.168.0.1',
];

// 扩展扫描池
const EXTENDED_TARGETS = (() => {
  const hosts = [2, 3, 10, 20, 30, 40, 50, 138, 150, 200,
                 111, 112, 113, 114, 115, 116, 117, 118, 119, 120];
  const subnets = ['192.168.1.', '192.168.0.', '192.168.31.', '192.168.2.'];
  const result = [];
  for (const s of subnets) {
    for (const h of hosts) {
      const ip = s + h;
      if (!PRIORITY_TARGETS.includes(ip)) result.push(ip);
    }
  }
  return result;
})();

const ALL_TARGETS = [...PRIORITY_TARGETS, ...EXTENDED_TARGETS];

/**
 * 扫描局域网内的康复监测服务器
 * @param {number} timeout 总超时（ms），默认 6000
 * @returns {Promise<Array<{ip, port, hostname}>>}
 */
function scanLan(timeout = 6000) {
  return new Promise((resolve) => {
    const found = [];
    const totalDeadline = Date.now() + timeout;
    let idx = 0;
    let running = 0;
    let stopped = false;

    const finish = () => {
      if (stopped) return;
      stopped = true;
      resolve(found);
    };

    const tryNext = () => {
      if (stopped) return;
      if (Date.now() >= totalDeadline) {
        if (running === 0) finish();
        return;
      }

      // 每次发 8 个并发请求，500ms 超时
      const CONCURRENCY = 8;
      while (running < CONCURRENCY && idx < ALL_TARGETS.length) {
        const ip = ALL_TARGETS[idx++];
        running++;
        probeIP(ip);
      }

      if (running === 0) finish();
    };

    const probeIP = (ip) => {
      wx.request({
        url: `http://${ip}:5000${HEALTH_PATH}`,
        method: 'GET',
        timeout: 500,
        success: (res) => {
          if (stopped) return;
          if (res.statusCode === 200 && res.data && res.data.code === 0) {
            const data = res.data.data || {};
            found.push({
              ip,
              port: 5000,
              hostname: data.hostname || '',
            });
            finish();
          }
        },
        fail: () => {},
        complete: () => {
          running--;
          if (!stopped) tryNext();
        },
      });
    };

    tryNext();
  });
}

module.exports = { scanLan };
