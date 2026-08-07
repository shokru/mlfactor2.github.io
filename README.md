# MyST Site Deployment for Railway

This package deploys your MyST site on Railway using a proxy wrapper that solves the localhost binding issue.

## How It Works

MyST's server only binds to `localhost`, which doesn't work on cloud platforms. This solution:

1. Starts MyST on `localhost:3000` (internal)
2. Runs a proxy server on `0.0.0.0:$PORT` (Railway's assigned port)
3. Forwards all HTTP and WebSocket traffic to MyST

This preserves MyST's full dynamic functionality including live reload.

## Files Included

| File | Purpose |
|------|---------|
| `package.json` | Dependencies and scripts |
| `server.js` | Proxy wrapper that bridges MyST to the public internet |
| `railway.json` | Railway-specific configuration |
| `.gitignore` | Ignores build artifacts |

## Deployment Instructions

### Step 1: Add Files to Your Repository

Copy these files to your repository root (alongside your existing MyST content like `myst.yml`, markdown files, etc.)

### Step 2: Deploy to Railway

**Option A: Via GitHub (Recommended)**

1. Push your changes to GitHub
2. Go to [railway.app](https://railway.app) and sign in
3. Click **"New Project"** → **"Deploy from GitHub repo"**
4. Select your repository
5. Railway will auto-detect and deploy

**Option B: Via Railway CLI**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize and deploy
railway init
railway up

# Get your public URL
railway domain
```

### Step 3: Verify Deployment

1. Go to your Railway dashboard
2. Click on your service
3. Check the **Deployments** tab for logs
4. Click the generated domain to view your site

## Environment Variables (Optional)

Railway automatically sets `PORT`. No additional environment variables are required.

If you need to customize, you can add these in Railway's dashboard under **Variables**:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | (set by Railway) | Public-facing port |

## Local Development

```bash
# Install dependencies
npm install

# Run MyST directly (recommended for local dev)
npm run dev

# Or test the proxy wrapper locally
npm start
```

## Troubleshooting

### "502 Bad Gateway" or "MyST server is starting up"

This is normal for the first few seconds after deployment. MyST needs time to initialize. Wait 10-15 seconds and refresh.

### Build fails

- Make sure you have a valid `myst.yml` in your repository
- Check that your markdown files are valid

### Site loads but pages are missing

- Verify your `myst.yml` configuration
- Check that all referenced files exist

### WebSocket errors in console

These are usually harmless in production. Live reload is primarily for development.

## Project Structure

Your repository should look like this:

```
your-repo/
├── package.json      # From this package
├── server.js         # From this package
├── railway.json      # From this package
├── .gitignore        # From this package
├── myst.yml          # Your MyST config
├── index.md          # Your content
├── chapter1.md       # Your content
└── ...               # Other MyST content
```

## Resources

- [Railway Documentation](https://docs.railway.app)
- [MyST Documentation](https://mystmd.org)
- [Railway Node.js Guide](https://docs.railway.app/guides/nodejs)
