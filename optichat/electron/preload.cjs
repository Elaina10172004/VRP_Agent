const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('desktopApp', {
  platform: process.platform,
  versions: {
    chrome: process.versions.chrome,
    electron: process.versions.electron,
    node: process.versions.node,
  },
  settings: {
    get: () => ipcRenderer.invoke('settings:get'),
    save: (settings) => ipcRenderer.invoke('settings:set', settings),
  },
  files: {
    openInstance: () => ipcRenderer.invoke('files:open-instance'),
    openInstances: () => ipcRenderer.invoke('files:open-instances'),
    saveJson: (request) => ipcRenderer.invoke('files:save-json', request),
  },
  solver: {
    ingestFile: (request) => ipcRenderer.invoke('solver:ingest-file', request),
    cancelIngest: (requestId) => ipcRenderer.invoke('solver:cancel-ingest', requestId),
    solve: (request) => ipcRenderer.invoke('solver:solve', request),
    onProgress: (callback) => {
      const listener = (_event, payload) => callback(payload);
      ipcRenderer.on('solver:progress', listener);
      return () => ipcRenderer.removeListener('solver:progress', listener);
    },
  },
});
