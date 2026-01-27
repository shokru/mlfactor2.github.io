/**
 * MyST Server Wrapper for Railway/Cloud Deployment
 * 
 * This script solves the problem where MyST only binds to localhost.
 * It starts MyST on localhost, then creates an HTTP proxy that binds
 * to 0.0.0.0 (all interfaces) so Railway can route traffic to it.
 */

const http = require('http');
const httpProxy = require('http-proxy');
const { spawn } = require('child_process');

// Railway provides PORT env var; MyST will use its default (3000)
const PUBLIC_PORT = process.env.PORT || 8080;
const MYST_PORT = 3000;

console.log(`Starting MyST deployment wrapper...`);
console.log(`Public port: ${PUBLIC_PORT}`);
console.log(`MyST internal port: ${MYST_PORT}`);

// Start MyST in the background
const mystProcess = spawn('npx', ['myst', 'start', '--port', String(MYST_PORT)], {
  stdio: ['ignore', 'pipe', 'pipe'],
  shell: true
});

mystProcess.stdout.on('data', (data) => {
  console.log(`[MyST] ${data.toString().trim()}`);
});

mystProcess.stderr.on('data', (data) => {
  console.error(`[MyST Error] ${data.toString().trim()}`);
});

mystProcess.on('error', (err) => {
  console.error('Failed to start MyST:', err);
  process.exit(1);
});

mystProcess.on('close', (code) => {
  console.log(`MyST process exited with code ${code}`);
  process.exit(code);
});

// Create proxy server
const proxy = httpProxy.createProxyServer({
  target: `http://localhost:${MYST_PORT}`,
  ws: true, // Enable WebSocket proxying for live reload
});

proxy.on('error', (err, req, res) => {
  console.error('Proxy error:', err.message);
  if (res.writeHead) {
    res.writeHead(502, { 'Content-Type': 'text/plain' });
    res.end('MyST server is starting up, please refresh in a moment...');
  }
});

// Create HTTP server that binds to all interfaces
const server = http.createServer((req, res) => {
  proxy.web(req, res);
});

// Handle WebSocket upgrades (for MyST live reload)
server.on('upgrade', (req, socket, head) => {
  proxy.ws(req, socket, head);
});

// Wait a bit for MyST to start, then start the proxy
setTimeout(() => {
  server.listen(PUBLIC_PORT, '0.0.0.0', () => {
    console.log(`\n✅ Proxy server listening on 0.0.0.0:${PUBLIC_PORT}`);
    console.log(`   Forwarding to MyST at localhost:${MYST_PORT}`);
    console.log(`\n🌐 Your site should be accessible now!\n`);
  });
}, 3000); // Give MyST 3 seconds to start

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('Received SIGTERM, shutting down...');
  mystProcess.kill();
  server.close();
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('Received SIGINT, shutting down...');
  mystProcess.kill();
  server.close();
  process.exit(0);
});
