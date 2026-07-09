/**
 * 局域网 UDP 服务发现
 * 
 * PC 服务端每 3 秒广播 server 地址到 255.255.255.255:5003，
 * 小程序监听 UDP 包自动获取服务端 IP。
 */

const DISCOVERY_PORT = 5003;
const SCAN_TIMEOUT = 2500;  // 扫描超时（ms），需 > 广播间隔 3s？不，收一个就够了

/**
 * 扫描局域网内的康复监测服务器
 * @param {number} timeout 扫描超时（ms），默认 2500
 * @returns {Promise<Array<{ip, port, ws_port, hostname}>>}
 */
function scanLan(timeout = SCAN_TIMEOUT) {
  return new Promise((resolve) => {
    const servers = [];
    const seen = new Set();
    let socket = null;
    let timer = null;

    const finish = () => {
      if (timer) clearTimeout(timer);
      if (socket) {
        try { socket.close(); } catch (e) {}
      }
      resolve(servers);
    };

    // 超时兜底
    timer = setTimeout(finish, timeout);

    try {
      socket = wx.createUDPSocket();
      const port = socket.bind(DISCOVERY_PORT);

      socket.onError(() => finish());

      socket.onMessage((res) => {
        try {
          // res.message 可能是 ArrayBuffer 或 string
          let text = '';
          if (typeof res.message === 'string') {
            text = res.message;
          } else if (res.message instanceof ArrayBuffer) {
            const buf = new Uint8Array(res.message);
            text = String.fromCharCode.apply(null, buf);
          } else {
            return;
          }

          const data = JSON.parse(text);
          if (data.type === 'rehab_server' && data.ip) {
            const key = `${data.ip}:${data.port}`;
            if (!seen.has(key)) {
              seen.add(key);
              servers.push({
                ip: data.ip,
                port: data.port || 5000,
                ws_port: data.ws_port || 5001,
                hostname: data.hostname || '',
              });
            }
          }
        } catch (e) {
          // 忽略解析失败的包
        }
      });

      // 端口绑定成功
      port.onListening(() => {
        // 已开始监听，等待超时
      });

    } catch (e) {
      // UDP socket 创建失败，直接返回空
      finish();
    }
  });
}

module.exports = { scanLan };
