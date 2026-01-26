/**
 * MyST Server Wrapper for Railway/Cloud Deployment (v2)
 * 
 * This script solves two problems:
 * 1. MyST only binds to localhost - we proxy to make it publicly accessible
 * 2. MyST generates localhost URLs for assets - we rewrite them
 */

const http = require('http');
const httpProxy = require('http-proxy');
const { spawn } = require('child_process');

// Railway provides PORT env var; MyST will use its default ports
const PUBLIC_PORT = process.env.PORT || 8080;
const MYST_CONTENT_PORT = 3100;  // MyST content server
const MYST_THEME_PORT = 3000;    // MyST theme server

console.log(`Starting MyST deployment wrapper v2...`);
console.log(`Public port: ${PUBLIC_PORT}`);
console.log(`MyST theme port: ${MYST_THEME_PORT}`);
console.log(`MyST content port: ${MYST_CONTENT_PORT}`);

// Start MyST in the background
const mystProcess = spawn('npx', ['myst', 'start'], {
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

// Create proxy for the theme server (main site)
const themeProxy = httpProxy.createProxyServer({
  target: `http://localhost:${MYST_THEME_PORT}`,
  ws: true,
  selfHandleResponse: true  // We'll handle the response to rewrite URLs
});

// Create proxy for content server (API/assets)
const contentProxy = httpProxy.createProxyServer({
  target: `http://localhost:${MYST_CONTENT_PORT}`,
  selfHandleResponse: true
});

// Function to rewrite localhost URLs in response body
function rewriteBody(body, req) {
  if (!body) return body;
  
  const host = req.headers.host || 'localhost';
  const protocol = req.headers['x-forwarded-proto'] || 'https';
  const publicUrl = `${protocol}://${host}`;
  
  // Replace all localhost references with the public URL
  let rewritten = body
    .replace(/http:\/\/localhost:3100/g, publicUrl)
    .replace(/http:\/\/localhost:3000/g, publicUrl)
    .replace(/localhost:3100/g, host)
    .replace(/localhost:3000/g, host);
  
  return rewritten;
}

// Handle theme proxy responses
themeProxy.on('proxyRes', (proxyRes, req, res) => {
  const contentType = proxyRes.headers['content-type'] || '';
  
  // Copy headers
  Object.keys(proxyRes.headers).forEach(key => {
    // Skip content-length as we might modify the body
    if (key.toLowerCase() !== 'content-length') {
      res.setHeader(key, proxyRes.headers[key]);
    }
  });
  res.statusCode = proxyRes.statusCode;
  
  // Only rewrite HTML and JSON responses
  if (contentType.includes('text/html') || contentType.includes('application/json')) {
    let body = [];
    proxyRes.on('data', chunk => body.push(chunk));
    proxyRes.on('end', () => {
      const original = Buffer.concat(body).toString('utf8');
      const rewritten = rewriteBody(original, req);
      res.end(rewritten);
    });
  } else {
    // For other content types (images, CSS, JS), pass through directly
    proxyRes.pipe(res);
  }
});

// Handle content proxy responses (same logic)
contentProxy.on('proxyRes', (proxyRes, req, res) => {
  const contentType = proxyRes.headers['content-type'] || '';
  
  Object.keys(proxyRes.headers).forEach(key => {
    if (key.toLowerCase() !== 'content-length') {
      res.setHeader(key, proxyRes.headers[key]);
    }
  });
  res.statusCode = proxyRes.statusCode;
  
  if (contentType.includes('text/html') || contentType.includes('application/json')) {
    let body = [];
    proxyRes.on('data', chunk => body.push(chunk));
    proxyRes.on('end', () => {
      const original = Buffer.concat(body).toString('utf8');
      const rewritten = rewriteBody(original, req);
      res.end(rewritten);
    });
  } else {
    proxyRes.pipe(res);
  }
});

// Error handlers
themeProxy.on('error', (err, req, res) => {
  console.error('Theme proxy error:', err.message);
  if (res.writeHead) {
    res.writeHead(502, { 'Content-Type': 'text/plain' });
    res.end('MyST server is starting up, please refresh in a moment...');
  }
});

contentProxy.on('error', (err, req, res) => {
  console.error('Content proxy error:', err.message);
  if (res.writeHead) {
    res.writeHead(502, { 'Content-Type': 'text/plain' });
    res.end('MyST content server is starting up, please refresh in a moment...');
  }
});

// Create HTTP server
const server = http.createServer((req, res) => {
  // Route to appropriate proxy based on path
  // MyST content server handles /config.json and /content/ paths
  if (req.url.startsWith('/config.json') || req.url.startsWith('/content/')) {
    contentProxy.web(req, res);
  } else {
    themeProxy.web(req, res);
  }
});

// Handle WebSocket upgrades
server.on('upgrade', (req, socket, head) => {
  themeProxy.ws(req, socket, head);
});

// Wait for MyST to start, then start the proxy
setTimeout(() => {
  server.listen(PUBLIC_PORT, '0.0.0.0', () => {
    console.log(`\n✅ Proxy server listening on 0.0.0.0:${PUBLIC_PORT}`);
    console.log(`   Rewriting localhost URLs to public domain`);
    console.log(`\n🌐 Your site should be accessible now!\n`);
  });
}, 5000); // Give MyST 5 seconds to start both servers

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
