# VRP Agent Desktop

Electron + React frontend for the VRP agent.

## Scripts

Install dependencies:

```bash
npm install
```

Start Vite only:

```bash
npm run dev
```

Start Electron in development mode:

```bash
npm run desktop:dev
```

Build and launch the packaged desktop shell:

```bash
npm run desktop:start
```

## Notes

- OpenAI-compatible settings are configured inside the desktop app.
- The frontend talks to the Python backend through Electron IPC.
- The recommended root entrypoint for normal use is `../start.bat`.
