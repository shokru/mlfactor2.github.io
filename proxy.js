const http = require('http');
const httpProxy = require('http-proxy');
const { spawn } = require('child_process');

const PORT = process.env.PORT || 10000;
const MYST_PORT = 3000;

// Start myst server
console.log('Starting myst server...');
const myst = spawn('myst', ['start', '--port', MYST_PORT.toString()], {
  stdio: 'inherit'
});

// Give myst a moment to start
setTimeout(() => {
  console.log('Starting proxy server...');
  
  // Create proxy server
  const proxy = httpProxy.createProxyServer({
    target: `http://127.0.0.1:${MYST_PORT}`,
    ws: true
  });

  // Create HTTP server that proxies to myst
  const server = http.createServer((req, res) => {
    proxy.web(req, res);
  });

  // Proxy WebSocket connections too
  server.on('upgrade', (req, socket, head) => {
    proxy.ws(req, socket, head);
  });

  server.listen(PORT, '0.0.0.0', () => {
    console.log(`Proxy server listening on 0.0.0.0:${PORT}`);
    console.log(`Forwarding to myst at 127.0.0.1:${MYST_PORT}`);
  });
}, 3000);
