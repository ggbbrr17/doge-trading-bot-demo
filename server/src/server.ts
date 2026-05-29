import express, { Request, Response } from 'express';
import * as http from 'http';
import * as WebSocket from 'ws';
import cors from 'cors';
import * as dotenv from 'dotenv';
import { TradingEngine } from './bot/tradingEngine';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const PORT = process.env.PORT || 5000;
const engine = new TradingEngine();

// Broadcast trading engine state payload to all connected clients
const broadcastState = () => {
  const payload = JSON.stringify(engine.getStatePayload());
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(payload);
    }
  });
};

// Hook trading engine updates to WebSocket broadcast
engine.registerUpdateCallback(() => {
  broadcastState();
});

// REST Endpoints
app.get('/api/state', (req: Request, res: Response) => {
  res.json(engine.getStatePayload());
});

app.post('/api/start', (req: Request, res: Response) => {
  try {
    engine.startBot();
    res.json({ success: true, message: 'AI core trading engine started successfully.' });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

app.post('/api/stop', (req: Request, res: Response) => {
  try {
    engine.stopBot();
    res.json({ success: true, message: 'AI core suspended successfully.' });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

app.post('/api/config', (req: Request, res: Response) => {
  try {
    const newConfig = req.body;
    engine.updateConfig(newConfig);
    res.json({ success: true, message: 'Configuration matrix successfully updated.' });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Start ticker processing immediately if config shows it was running previously
if (engine.getStatePayload().config.isRunning) {
  // Turn it off first to avoid double tickers
  engine.updateConfig({ isRunning: false });
}

// WebSocket connection handler
wss.on('connection', (ws) => {
  // Send current state on connection
  ws.send(JSON.stringify(engine.getStatePayload()));

  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message.toString());
      if (data.action === 'PING') {
        ws.send(JSON.stringify({ type: 'PONG' }));
      }
    } catch (e) {
      console.error('Error handling WebSocket message:', e);
    }
  });
});

// Start Server
server.listen(PORT, () => {
  console.log(`==================================================`);
  console.log(`🚀 DOGE/USDT AI Trading Engine server is active!`);
  console.log(`   REST API: http://localhost:${PORT}/api`);
  console.log(`   WebSockets: ws://localhost:${PORT}`);
  console.log(`==================================================`);
});
