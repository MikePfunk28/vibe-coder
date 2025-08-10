const { app, BrowserWindow, Menu, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');

let mainWindow;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
            enableRemoteModule: true
        },
        icon: path.join(__dirname, 'assets', 'icon.png'),
        title: 'AI IDE - VSCode Alternative with AI Features'
    });

    mainWindow.loadFile('index.html');
    
    // Open DevTools in development
    if (process.env.NODE_ENV === 'development') {
        mainWindow.webContents.openDevTools();
    }

    // Create menu
    createMenu();
}

function createMenu() {
    const template = [
        {
            label: 'File',
            submenu: [
                {
                    label: 'New File',
                    accelerator: 'CmdOrCtrl+N',
                    click: () => {
                        mainWindow.webContents.send('menu-new-file');
                    }
                },
                {
                    label: 'Open File',
                    accelerator: 'CmdOrCtrl+O',
                    click: async () => {
                        const result = await dialog.showOpenDialog(mainWindow, {
                            properties: ['openFile'],
                            filters: [
                                { name: 'All Files', extensions: ['*'] },
                                { name: 'JavaScript', extensions: ['js', 'jsx'] },
                                { name: 'TypeScript', extensions: ['ts', 'tsx'] },
                                { name: 'Python', extensions: ['py'] },
                                { name: 'HTML', extensions: ['html', 'htm'] },
                                { name: 'CSS', extensions: ['css'] },
                                { name: 'JSON', extensions: ['json'] }
                            ]
                        });
                        
                        if (!result.canceled) {
                            const filePath = result.filePaths[0];
                            const content = fs.readFileSync(filePath, 'utf8');
                            mainWindow.webContents.send('file-opened', { path: filePath, content });
                        }
                    }
                },
                {
                    label: 'Save',
                    accelerator: 'CmdOrCtrl+S',
                    click: () => {
                        mainWindow.webContents.send('menu-save');
                    }
                },
                { type: 'separator' },
                {
                    label: 'Exit',
                    accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
                    click: () => {
                        app.quit();
                    }
                }
            ]
        },
        {
            label: 'AI',
            submenu: [
                {
                    label: 'Generate Code',
                    accelerator: 'Ctrl+K',
                    click: () => {
                        mainWindow.webContents.send('ai-generate');
                    }
                },
                {
                    label: 'Open AI Chat',
                    accelerator: 'Ctrl+L',
                    click: () => {
                        mainWindow.webContents.send('ai-chat');
                    }
                },
                {
                    label: 'Explain Code',
                    accelerator: 'Ctrl+E',
                    click: () => {
                        mainWindow.webContents.send('ai-explain');
                    }
                }
            ]
        },
        {
            label: 'View',
            submenu: [
                { role: 'reload' },
                { role: 'forceReload' },
                { role: 'toggleDevTools' },
                { type: 'separator' },
                { role: 'resetZoom' },
                { role: 'zoomIn' },
                { role: 'zoomOut' },
                { type: 'separator' },
                { role: 'togglefullscreen' }
            ]
        }
    ];

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
}

// Handle file save
ipcMain.handle('save-file', async (event, { path, content }) => {
    try {
        if (path) {
            fs.writeFileSync(path, content);
            return { success: true };
        } else {
            const result = await dialog.showSaveDialog(mainWindow, {
                filters: [
                    { name: 'All Files', extensions: ['*'] },
                    { name: 'JavaScript', extensions: ['js'] },
                    { name: 'TypeScript', extensions: ['ts'] },
                    { name: 'Python', extensions: ['py'] },
                    { name: 'HTML', extensions: ['html'] },
                    { name: 'CSS', extensions: ['css'] },
                    { name: 'JSON', extensions: ['json'] }
                ]
            });
            
            if (!result.canceled) {
                fs.writeFileSync(result.filePath, content);
                return { success: true, path: result.filePath };
            }
        }
        return { success: false };
    } catch (error) {
        return { success: false, error: error.message };
    }
});

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});
