const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Serve static files from _build/site
app.use(express.static(path.join(__dirname, '_build/site')));

// SPA fallback - serve index.html for all routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '_build/site', 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on port ${PORT}`);
});
