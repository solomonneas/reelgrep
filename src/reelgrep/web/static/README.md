# reelgrep web static assets

These files (`index.html`, `style.css`, `app.js`) ship inside the wheel via
`importlib.resources` and are mounted by the `reelgrep serve` command at the
web root. The frontend is pure vanilla HTML / CSS / JS (ES modules, no build
step, no framework, no CDN dependencies). It talks to the local backend over
`/api/*` JSON endpoints and proxies image / video / manifest files via the
`/file?path=` route. Open the served URL in any modern browser.
