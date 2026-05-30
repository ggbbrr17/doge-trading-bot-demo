import * as dotenv from 'dotenv';
dotenv.config();

import express, { Request, Response } from 'express';
import * as http from 'http';
import * as WebSocket from 'ws';
import cors from 'cors';
import mongoose from 'mongoose';
import * as path from 'path';
import { TradingEngine } from './bot/tradingEngine';
import { saveTradeToHistory } from './persistence';

// Auto-restart on unhandled errors: Exiting with code 1 signals Render to restart the container
process.on('uncaughtException', (error) => {
  console.error('💥 CRITICAL UNCAUGHT EXCEPTION:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('💥 UNHANDLED REJECTION AT:', promise, 'REASON:', reason);
  process.exit(1);
});

const app = express();
app.use(cors());
app.use(express.json());

// Serve static files from the React client app
const clientDistPath = path.join(__dirname, '../../client/dist');
app.use(express.static(clientDistPath));

const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const PORT = process.env.PORT || 5000;
const engine = new TradingEngine();

// Registro interno para evitar duplicados en el archivo JSON
const processedTradeStates = new Set<string>();

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
  const state = engine.getStatePayload();

  // Monitorear trades para persistencia automática
  state.trades.forEach((trade: any) => {
    const stateKey = `${trade.id}-${trade.status}`;
    if (!processedTradeStates.has(stateKey)) {
      saveTradeToHistory(trade);
      processedTradeStates.add(stateKey);
    }
  });

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

app.post('/api/send-telegram-test', (req: Request, res: Response) => {
  try {
    const { token, chatId } = req.body;
    if (!token || !chatId) {
      return res.status(400).json({ success: false, error: 'Missing token or chatId' });
    }
    engine.sendTelegramTest(token, chatId);
    res.json({ success: true, message: 'Test message sent to Telegram.' });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

app.post('/api/send-telegram-summary', async (req: Request, res: Response) => {
  try {
    await engine.sendSummaryNow();
    res.json({ success: true, message: 'Resumen enviado a Telegram exitosamente.' });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Fallback to serve index.html for SPA routing
app.get('*', (req: Request, res: Response) => {
  res.sendFile(path.join(clientDistPath, 'index.html'));
});

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
const MONGODB_URI = process.env.MONGODB_URI || '';

if (!MONGODB_URI) {
  console.error('❌ CRITICAL: MONGODB_URI is not defined in environment variables.');
  process.exit(1);
}

mongoose.connect(MONGODB_URI, {
  serverSelectionTimeoutMS: 10000, // Aumentamos el tiempo de espera
  heartbeatFrequencyMS: 2000,      // Mantiene la conexión viva
  socketTimeoutMS: 45000,
})
  .then(async () => {
    console.log('✨ Connected to MongoDB Atlas');

    // Initialize engine (loads state and trades from DB)
    await engine.initialize();

    server.listen(PORT, () => {
      console.log(`==================================================`);
      console.log(`🚀 DOGE/USDT AI Trading Engine server is active!`);
      console.log(`   REST API: http://localhost:${PORT}/api`);
      console.log(`   WebSockets: ws://localhost:${PORT}`);
      console.log(`==================================================`);

      // Self-ping logic
      const selfPingUrl = process.env.RENDER_EXTERNAL_URL || process.env.SELF_PING_URL;
      if (selfPingUrl) {
        setInterval(async () => {
          try {
            const response = await fetch(selfPingUrl);
            console.log(`[Self-Ping] Heartbeat: ${response.status}`);
          } catch (e: any) {
            console.error(`[Self-Ping] Heartbeat failed: ${e.message}`);
          }
        }, 5 * 60 * 1000);
      }
    });
  })
  .catch((err: Error) => {
    console.error('❌ MongoDB Connection Error:', err);
    process.exit(1);
  });
