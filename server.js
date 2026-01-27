/**
 * MyST Server Wrapper for Railway/Cloud Deployment
 * 
 * This script solves the problem where MyST only binds to localhost.
 * It starts MyST on localhost, then creates an HTTP proxy that binds
 * to 0.0.0.0 (all interfaces) so Railway can route traffic to it.
 * 
 * Additionally, it rewrites localhost URLs in responses to use the
 * public Railway URL so images and assets load correctly.
 */

const http = require('http');
const httpProxy = require('http-proxy');
const { spawn } = require('child_process');
const zlib = require('zlib');

// Railway provides PORT env var; MyST will use its default (3000)
const PUBLIC_PORT = process.env.PORT || 8080;
const MYST_PORT = 3100;

// Get the public URL from Railway environment
// Railway sets RAILWAY_PUBLIC_DOMAIN automatically
const PUBLIC_HOST = process.env.RAILWAY_PUBLIC_DOMAIN 
  || process.env.PUBLIC_URL 
  || null;

console.log(`Starting MyST deployment wrapper...`);
console.log(`Public port: ${PUBLIC_PORT}`);
console.log(`MyST internal port: ${MYST_PORT}`);
console.log(`Public host: ${PUBLIC_HOST || '(not set - URL rewriting disabled)'}`);

// Start MyST in the background
const mystProcess = spawn('npx', ['myst', 'start', '--port', String(MYST_PORT)], {
  stdio: ['ignore', 'pipe', 'pipe'],
  shell: true,
  env: { ...process.env, HOST: undefined }
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

// Function to rewrite localhost URLs in content
function rewriteUrls(body, publicHost) {
  if (!publicHost) return body;
  
  // Patterns to replace localhost references
  const patterns = [
    // http://localhost:3100 -> https://public-domain
    [/http:\/\/localhost:3100/g, `https://${publicHost}`],
    // http://localhost:${MYST_PORT} -> https://public-domain  
    [new RegExp(`http://localhost:${MYST_PORT}`, 'g'), `https://${publicHost}`],
    // //localhost:3100 -> //public-domain
    [/\/\/localhost:3100/g, `//${publicHost}`],
    // Also handle 127.0.0.1
    [/http:\/\/127\.0\.0\.1:3100/g, `https://${publicHost}`],
    [new RegExp(`http://127\\.0\\.0\\.1:${MYST_PORT}`, 'g'), `https://${publicHost}`],
    // Catch any other common MyST ports just in case
    [/http:\/\/localhost:3000/g, `https://${publicHost}`],
    [/\/\/localhost:3000/g, `//${publicHost}`],
  ];
  
  let result = body;
  for (const [pattern, replacement] of patterns) {
    result = result.replace(pattern, replacement);
  }
  return result;
}

// Create proxy server with selfHandleResponse for URL rewriting
const proxy = httpProxy.createProxyServer({
  target: `http://localhost:${MYST_PORT}`,
  ws: true,
  selfHandleResponse: true, // We'll handle the response ourselves
});

proxy.on('proxyRes', (proxyRes, req, res) => {
  const contentType = proxyRes.headers['content-type'] || '';
  const contentEncoding = proxyRes.headers['content-encoding'];
  
  // Only rewrite HTML, CSS, and JavaScript responses
  const shouldRewrite = PUBLIC_HOST && (
    contentType.includes('text/html') ||
    contentType.includes('text/css') ||
    contentType.includes('application/javascript') ||
    contentType.includes('application/json')
  );
  
  if (!shouldRewrite) {
    // Pass through non-text responses unchanged (images, fonts, etc.)
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res);
    return;
  }
  
  // Collect the response body
  const chunks = [];
  proxyRes.on('data', (chunk) => chunks.push(chunk));
  proxyRes.on('end', () => {
    let body = Buffer.concat(chunks);
    
    // Handle gzip/deflate compressed responses
    const decompress = (buffer, callback) => {
      if (contentEncoding === 'gzip') {
        zlib.gunzip(buffer, callback);
      } else if (contentEncoding === 'deflate') {
        zlib.inflate(buffer, callback);
      } else if (contentEncoding === 'br') {
        zlib.brotliDecompress(buffer, callback);
      } else {
        callback(null, buffer);
      }
    };
    
    decompress(body, (err, decompressed) => {
      if (err) {
        console.error('Decompression error:', err);
        res.writeHead(proxyRes.statusCode, proxyRes.headers);
        res.end(body);
        return;
      }
      
      // Rewrite URLs in the response
      let content = decompressed.toString('utf8');
      content = rewriteUrls(content, PUBLIC_HOST);
      
      // Remove content-encoding since we're sending uncompressed
      const headers = { ...proxyRes.headers };
      delete headers['content-encoding'];
      delete headers['content-length']; // Length will change
      
      res.writeHead(proxyRes.statusCode, headers);
      res.end(content);
    });
  });
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
    if (PUBLIC_HOST) {
      console.log(`   Rewriting URLs to https://${PUBLIC_HOST}`);
    }
    console.log(`\n🌐 Your site should be accessible now!\n`);
  });
}, 3000);

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
