/**
 * MyST Server Wrapper for Railway/Cloud Deployment
 * 
 * MyST runs TWO servers:
 * - Port 3000: Theme server (the rendered website)
 * - Port 3100: Content server (JSON API, images, assets)
 * 
 * This proxy routes requests appropriately and rewrites localhost URLs.
 */

const http = require('http');
const httpProxy = require('http-proxy');
const { spawn } = require('child_process');
const zlib = require('zlib');

const PUBLIC_PORT = process.env.PORT || 8080;
const THEME_PORT = 3000;    // MyST theme server
const CONTENT_PORT = 3100;  // MyST content server

// Get the public URL from Railway environment
const PUBLIC_HOST = process.env.RAILWAY_PUBLIC_DOMAIN 
  || process.env.PUBLIC_URL 
  || null;

console.log(`Starting MyST deployment wrapper...`);
console.log(`Public port: ${PUBLIC_PORT}`);
console.log(`MyST theme port: ${THEME_PORT}`);
console.log(`MyST content port: ${CONTENT_PORT}`);
console.log(`Public host: ${PUBLIC_HOST || '(not set - URL rewriting disabled)'}`);

// Start MyST in the background
const mystProcess = spawn('npx', ['myst', 'start', '--port', String(THEME_PORT)], {
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
  
  // Replace all localhost references with public host
  const patterns = [
    // Content server URLs (port 3100)
    [/http:\/\/localhost:3100/g, `https://${publicHost}`],
    [/\/\/localhost:3100/g, `//${publicHost}`],
    [/http:\/\/127\.0\.0\.1:3100/g, `https://${publicHost}`],
    // Theme server URLs (port 3000)  
    [/http:\/\/localhost:3000/g, `https://${publicHost}`],
    [/\/\/localhost:3000/g, `//${publicHost}`],
    [/http:\/\/127\.0\.0\.1:3000/g, `https://${publicHost}`],
  ];
  
  let result = body;
  for (const [pattern, replacement] of patterns) {
    result = result.replace(pattern, replacement);
  }
  return result;
}

// Create proxy for theme server (with response rewriting)
const themeProxy = httpProxy.createProxyServer({
  target: `http://localhost:${THEME_PORT}`,
  ws: true,
  selfHandleResponse: true,
});

// Create proxy for content server (images, JSON)
const contentProxy = httpProxy.createProxyServer({
  target: `http://localhost:${CONTENT_PORT}`,
  ws: true,
  selfHandleResponse: false, // Pass through as-is for binary content
});

// Handle theme proxy responses (with URL rewriting)
themeProxy.on('proxyRes', (proxyRes, req, res) => {
  const contentType = proxyRes.headers['content-type'] || '';
  const contentEncoding = proxyRes.headers['content-encoding'];
  
  const shouldRewrite = PUBLIC_HOST && (
    contentType.includes('text/html') ||
    contentType.includes('text/css') ||
    contentType.includes('application/javascript') ||
    contentType.includes('application/json')
  );
  
  if (!shouldRewrite) {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res);
    return;
  }
  
  const chunks = [];
  proxyRes.on('data', (chunk) => chunks.push(chunk));
  proxyRes.on('end', () => {
    let body = Buffer.concat(chunks);
    
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
      
      let content = decompressed.toString('utf8');
      content = rewriteUrls(content, PUBLIC_HOST);
      
      const headers = { ...proxyRes.headers };
      delete headers['content-encoding'];
      delete headers['content-length'];
      
      res.writeHead(proxyRes.statusCode, headers);
      res.end(content);
    });
  });
});

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

// Determine which proxy to use based on the request path
function routeRequest(req) {
  const url = req.url || '';
  
  // Routes that go to the content server (port 3100)
  // These typically include images, thumbnails, and API content
  const contentPaths = [
    '/build/',        // Built content including images
    '/content/',      // Content API
    '/config.json',   // Site configuration
    '/myst.xref.json', // Cross-references
    '/objects.inv',   // Inventory file
  ];
  
  // Check if this is a content server request
  for (const path of contentPaths) {
    if (url.startsWith(path)) {
      return 'content';
    }
  }
  
  // Check for image/static file extensions that might be served from content server
  const contentExtensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.ico', '.pdf'];
  for (const ext of contentExtensions) {
    if (url.toLowerCase().includes(ext)) {
      return 'content';
    }
  }
  
  // Everything else goes to theme server
  return 'theme';
}

// Create HTTP server
const server = http.createServer((req, res) => {
  const target = routeRequest(req);
  
  if (target === 'content') {
    console.log(`[Content] ${req.method} ${req.url}`);
    contentProxy.web(req, res);
  } else {
    console.log(`[Theme] ${req.method} ${req.url}`);
    themeProxy.web(req, res);
  }
});

// Handle WebSocket upgrades
server.on('upgrade', (req, socket, head) => {
  const target = routeRequest(req);
  if (target === 'content') {
    contentProxy.ws(req, socket, head);
  } else {
    themeProxy.ws(req, socket, head);
  }
});

// Wait for MyST to start
setTimeout(() => {
  server.listen(PUBLIC_PORT, '0.0.0.0', () => {
    console.log(`\n✅ Proxy server listening on 0.0.0.0:${PUBLIC_PORT}`);
    console.log(`   Theme requests → localhost:${THEME_PORT}`);
    console.log(`   Content requests → localhost:${CONTENT_PORT}`);
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
