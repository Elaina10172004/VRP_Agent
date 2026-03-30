import http from 'node:http';
import path from 'node:path';
import process from 'node:process';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const appDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const shellCommand = process.platform === 'win32' ? process.env.ComSpec || 'cmd.exe' : null;
const electronCommand = process.platform === 'win32'
  ? path.join(appDir, 'node_modules', 'electron', 'dist', 'electron.exe')
  : path.join(appDir, 'node_modules', '.bin', 'electron');
const electronEntry = path.join(appDir, 'electron', 'main.cjs');
const children = [];
let shuttingDown = false;

function spawnTask(label, command, args, env = process.env) {
  const child = spawn(command, args, {
    cwd: appDir,
    stdio: 'inherit',
    windowsHide: false,
    env,
  });

  children.push(child);
  child.on('exit', (code, signal) => {
    if (shuttingDown) {
      return;
    }
    const reason = signal ? `signal ${signal}` : `code ${code ?? 0}`;
    console.error(`[desktop] ${label} exited with ${reason}`);
    shutdown(code ?? 1);
  });
}

function startNpmTask(label, npmArgs) {
  if (process.platform === 'win32') {
    spawnTask(label, shellCommand, ['/d', '/s', '/c', npmCommand, ...npmArgs]);
    return;
  }
  spawnTask(label, npmCommand, npmArgs);
}

function shutdown(code = 0) {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;
  for (const child of children) {
    if (!child.pid) {
      continue;
    }
    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', String(child.pid), '/t', '/f'], {
        stdio: 'ignore',
        windowsHide: true,
      });
    } else {
      try {
        process.kill(child.pid, 'SIGTERM');
      } catch {
        // ignore shutdown race
      }
    }
  }
  setTimeout(() => process.exit(code), 400);
}

function waitForHttp(url, timeoutMs = 120000) {
  const startedAt = Date.now();
  return new Promise((resolve, reject) => {
    const probe = () => {
      const req = http.get(url, (res) => {
        res.resume();
        resolve(res.statusCode ?? 200);
      });
      req.on('error', () => {
        if (Date.now() - startedAt >= timeoutMs) {
          reject(new Error(`Timed out waiting for ${url}`));
          return;
        }
        setTimeout(probe, 1000);
      });
    };
    probe();
  });
}

process.on('SIGINT', () => shutdown(0));
process.on('SIGTERM', () => shutdown(0));

console.log('[desktop] starting vite dev server...');
startNpmTask('vite', ['run', 'dev']);

try {
  await waitForHttp('http://127.0.0.1:3000');
  console.log('[desktop] frontend is reachable on http://127.0.0.1:3000');
} catch (error) {
  console.error(`[desktop] ${error instanceof Error ? error.message : 'failed to start vite dev server'}`);
  shutdown(1);
}

if (!shuttingDown) {
  spawnTask('electron', electronCommand, [electronEntry], {
    ...process.env,
    OPTICHAT_DEV: 'true',
    OPTICHAT_START_URL: 'http://127.0.0.1:3000',
  });
}

setInterval(() => {
  if (!shuttingDown) {
    return;
  }
}, 1000);
