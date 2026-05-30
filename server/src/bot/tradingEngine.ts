import mongoose from 'mongoose';
import { BinanceClient } from '../utils/binanceClient';
import { CustomNeuralNetwork } from './aiModel';
import { calculateIndicators, StrategyManager, StrategySignal, TechnicalIndicators } from './strategies';
import { EvolutionEngine } from './evolutionEngine';
import { GemmaService, GemmaSignal } from './gemmaService';
import { hmmService, HMMResult } from './hmmService';
import { TradeModel } from '../persistence';
import { OrderBookSignal } from '../orderBookSensor';

// Esquema para guardar la configuración y stats del bot
const BotStateSchema = new mongoose.Schema({
  key: { type: String, default: 'current_state' },
  config: Object,
  stats: Object,
  updatedAt: { type: Date, default: Date.now }
});
const BotStateModel = mongoose.model('BotState', BotStateSchema);

export interface Trade {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  type: 'TESTNET' | 'REAL';
  price: number;
  quantity: number;
  amount: number;
  timestamp: number;
  status: 'OPEN' | 'CLOSED';
  pnl?: number;      // PnL in USDT
  pnlPercent?: number; // PnL in %
  exitPrice?: number;
  exitTimestamp?: number;
  reason?: string;
  targetSL?: number;
  targetTP?: number;
  highestPrice?: number;
  lowestPrice?: number;
  isBreakevenActive?: boolean;
}

const MAX_RISK_PER_TRADE_PERCENT = 1.5; // Arriesgar máximo 1.5% del balance por operación

export interface BotStats {
  totalBalanceUSDT: number;
  dogeBalance: number;
  netProfitUSDT: number;
  winRatePercent: number;
  profitFactor: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  dailyPnL?: number;
  lastPnLReset?: string;
  lastTelegramUpdateId?: number;
}

export interface BotConfig {
  mode: 'TESTNET' | 'REAL';
  isRunning: boolean;
  strategy: 'AI_UNIFIED';
  tradeSizeUSDT: number;
  geminiApiKey: string;
  binanceApiKey: string;
  binanceApiSecret: string;
  gridLayers: number;
  marketType: 'SPOT' | 'FUTURES';
  leverage: number;
  dailyProfitTarget?: number; // Meta de ganancia diaria en USDT
  telegramBotToken?: string;
  telegramChatId?: string;
}

export class TradingEngine {
  private config: BotConfig;
  private stats: BotStats;
  private trades: Trade[] = [];
  private logQueue: string[] = [];

  private pricesBuffer: number[] = [];
  private candles: { time: number; open: number; high: number; low: number; close: number; volume: number }[] = [];

  private neuralNet: CustomNeuralNetwork;
  private strategyManager: StrategyManager;
  private evolutionEngine: EvolutionEngine;
  private gemmaService: GemmaService;
  private cachedGemmaSignal: GemmaSignal | null = null;
  private cachedOrderBook: OrderBookSignal | null = null;
  private lastGemmaFetchTime = 0;
  private isGemmaFetching = false;
  private binanceClient: BinanceClient | null = null;

  private currentRegime: HMMResult | null = null;
  private lastHmmFetchTime = 0;
  private isHmmFetching = false;

  private telegramIntervalId: NodeJS.Timeout | null = null;
  private lastSummaryTime = 0;
  private activeIntervalId: NodeJS.Timeout | null = null;
  private onUpdateCallback: (() => void) | null = null;
  private tickCount = 0;

  constructor() {
    this.neuralNet = new CustomNeuralNetwork();
    this.strategyManager = new StrategyManager(this.neuralNet);
    this.evolutionEngine = new EvolutionEngine();
    this.gemmaService = new GemmaService();

    // Initial Defaults
    this.config = {
      mode: 'TESTNET',
      isRunning: false,
      strategy: 'AI_UNIFIED',
      tradeSizeUSDT: Number(process.env.TRADE_SIZE_USDT) || 50,
      geminiApiKey: process.env.GEMINI_API_KEY || '',
      binanceApiKey: process.env.BINANCE_API_KEY || '',
      binanceApiSecret: process.env.BINANCE_API_SECRET || '',
      gridLayers: 3,
      marketType: (process.env.MARKET_TYPE as 'SPOT' | 'FUTURES') || 'SPOT',
      leverage: Number(process.env.LEVERAGE) || 5,
      telegramBotToken: process.env.TELEGRAM_BOT_TOKEN || '',
      telegramChatId: process.env.TELEGRAM_CHAT_ID || '',
    };

    this.stats = {
      totalBalanceUSDT: 10000.0,
      dogeBalance: 0.0,
      netProfitUSDT: 0.0,
      winRatePercent: 0.0,
      profitFactor: 0.0,
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      dailyPnL: 0,
      lastPnLReset: new Date().toISOString().split('T')[0]
    };

    // Nota: loadState ahora es asíncrono, se llama desde el inicio del servidor
  }

  public async initialize() {
    await this.loadState();
    await this.loadActiveTradesFromDb();
    this.log('Engine initialized with MongoDB state.');

    // Propagate loaded API key to AI services on startup
    if (this.config.geminiApiKey) {
      this.gemmaService.updateApiKey(this.config.geminiApiKey);
      this.evolutionEngine.updateApiKey(this.config.geminiApiKey);
    } else {
      this.log('⚠️ No Gemini API Key found in persistent state.');
    }

    // Auto-start si está configurado por entorno para asegurar 24/7
    if (process.env.AUTO_START === 'true') {
      setTimeout(() => this.startBot(), 5000);
    }

    this.initializeBinance();
    this.initializeTelegram();
    this.seedCandles();
  }

  private async loadActiveTradesFromDb() {
    try {
      const activeTrades = await TradeModel.find({ status: 'OPEN' });
      this.trades = activeTrades.map((t: any) => t.toObject() as any);
      this.log(`Loaded ${this.trades.length} active trades from MongoDB.`);
    } catch (e) {
      this.log('Failed to load active trades from DB.');
    }
  }

  // Set visual websocket update callback
  registerUpdateCallback(callback: () => void) {
    this.onUpdateCallback = callback;
  }

  private triggerUpdate() {
    if (this.onUpdateCallback) this.onUpdateCallback();
  }

  private log(message: string) {
    const timestamp = new Date().toLocaleTimeString();
    const formatted = `[${timestamp}] ${message}`;
    this.logQueue.unshift(formatted);
    if (this.logQueue.length > 100) this.logQueue.pop();
    console.log(formatted);
    this.triggerUpdate();
  }

  getLogs(): string[] {
    return this.logQueue;
  }

  // Pre-seed some candles for chart initialization and indicator warm-up
  private async seedCandles() {
    this.log('Initializing historical market feeds for DOGE/USDT...');
    try {
      // Fetch 100 recent 1-minute klines from public Binance (or use fallback generator if network fails)
      let rawKlines: any[] = [];
      try {
        const client = new BinanceClient({ apiKey: '', apiSecret: '', isTestnet: false, marketType: 'SPOT' });
        rawKlines = await client.getKlines('DOGEUSDT', '1m', 80);
      } catch (e) {
        this.log('Using simulated offline market stream generator.');
      }

      if (rawKlines && rawKlines.length > 0) {
        this.candles = rawKlines.map((k) => ({
          time: k[0] / 1000,
          open: parseFloat(k[1]),
          high: parseFloat(k[2]),
          low: parseFloat(k[3]),
          close: parseFloat(k[4]),
          volume: parseFloat(k[5]),
        }));
        this.pricesBuffer = this.candles.map((c) => c.close);
        this.log(`Successfully warmed indicators with ${this.candles.length} historical candles.`);
      } else {
        // Mock Generator Fallback
        let currentPrice = 0.42;
        let now = Math.floor(Date.now() / 1000) - 80 * 60;
        for (let i = 0; i < 80; i++) {
          const change = (Math.random() - 0.49) * 0.003 * currentPrice; // slight upwards bias
          const open = currentPrice;
          const close = currentPrice + change;
          const high = Math.max(open, close) + Math.random() * 0.001 * currentPrice;
          const low = Math.min(open, close) - Math.random() * 0.001 * currentPrice;
          this.candles.push({ time: now, open, high, low, close, volume: Math.random() * 100000 + 50000 });
          this.pricesBuffer.push(close);
          currentPrice = close;
          now += 60;
        }
        this.log('Seeded engine with high-fidelity synthetic market buffer.');
      }
    } catch (error) {
      this.log(`Error seeding candles: ${error}`);
    }
    this.triggerUpdate();
  }

  // Initialize/Update Binance connection
  initializeBinance() {
    if (this.config.binanceApiKey && this.config.binanceApiSecret) {
      this.binanceClient = new BinanceClient({
        apiKey: this.config.binanceApiKey,
        apiSecret: this.config.binanceApiSecret,
        isTestnet: this.config.mode === 'TESTNET',
        marketType: this.config.marketType,
      });
      this.log(`Binance Client authenticated in ${this.config.mode} mode (${this.config.marketType}).`);

      if (this.config.marketType === 'FUTURES') {
        this.binanceClient.setLeverage('DOGEUSDT', this.config.leverage)
          .then(() => this.log(`Binance leverage successfully configured to ${this.config.leverage}x.`))
          .catch((err) => this.log(`Warning: Failed to set Binance leverage to ${this.config.leverage}x: ${err.message}`));
      }
    } else {
      this.binanceClient = null;
      this.log(`WARNING: Binance keys are missing. Verification required for ${this.config.mode} mode.`);
    }
  }

  // Initialize/Update Telegram connection (just logs for now)
  initializeTelegram() {
    if (this.telegramIntervalId) {
      clearInterval(this.telegramIntervalId);
      this.telegramIntervalId = null;
    }

    if (this.config.telegramBotToken && this.config.telegramChatId) {
      this.log('Telegram notifications and commands enabled.');
      // Poll for new messages every 5 seconds
      this.telegramIntervalId = setInterval(() => this.handleTelegramUpdates(), 5000);
    } else {
      this.log('Telegram notifications disabled (missing token or chat ID).');
    }
  }

  private async handleTelegramUpdates() {
    const token = this.config.telegramBotToken;
    if (!token) return;

    try {
      const offset = this.stats.lastTelegramUpdateId ? this.stats.lastTelegramUpdateId + 1 : 0;
      const response = await fetch(`https://api.telegram.org/bot${token}/getUpdates?offset=${offset}&timeout=10`);

      if (!response.ok) return;

      const data = await response.json() as any;
      if (data.ok && data.result.length > 0) {
        for (const update of data.result) {
          this.stats.lastTelegramUpdateId = update.update_id;

          const message = update.message;
          if (!message || !message.text) continue;

          const text = message.text.toLowerCase().trim();
          const chatId = message.chat.id.toString();

          if (chatId === this.config.telegramChatId) {
            if (text.includes('como va todo') || text.includes('status') || text.includes('estado')) {
              this.log('📨 Telegram command: Full Status Report requested.');
              const report = await this.generateStatusReport();
              await this.sendTelegramMessage(report);
            } else if (text.includes('mercado') || text.includes('como va')) {
              this.log('📨 Telegram command: Market Summary request.');
              const summary = await this.gemmaService.generateMarketSummary();
              await this.sendTelegramMessage(`🤖 *Análisis de Mercado:* \n\n${summary}`);
            }
          }
        }
        this.saveState();
      }
    } catch (error: any) {
      // Fail silently to avoid clogging logs during polling
    }
  }

  // Load state from file
  private async loadState() {
    try {
      const state = await BotStateModel.findOne({ key: 'current_state' });
      if (state) {
        // Combinar configuración guardada, pero asegurar que no sobreescribimos con vacíos si hay env vars
        const savedConfig = state.config || {};

        // PRIORIDAD: Si el valor en DB es vacío, NO sobrescribir el valor de Render (process.env)
        Object.keys(savedConfig).forEach(key => {
          const val = (savedConfig as any)[key];
          if (val !== "" && val !== null && val !== undefined) {
            (this.config as any)[key] = val;
          }
        });

        if (state.stats) this.stats = { ...this.stats, ...state.stats };
      }

      // Refuerzo: Asegurar que si la DB no tenía nada, usamos las de Render
      this.config.geminiApiKey = this.config.geminiApiKey || process.env.GEMINI_API_KEY || '';
      this.config.binanceApiKey = this.config.binanceApiKey || process.env.BINANCE_API_KEY || '';
      this.config.binanceApiSecret = this.config.binanceApiSecret || process.env.BINANCE_API_SECRET || '';

      this.log('System state successfully loaded.');
    } catch (error) {
      this.log('Failed to load state from DB. Using defaults.');
    }
  }

  // Save state to file
  private async saveState() {
    try {
      await BotStateModel.findOneAndUpdate(
        { key: 'current_state' },
        { config: this.config, stats: this.stats, updatedAt: new Date() },
        { upsert: true }
      );
    } catch (error) {
      console.error('Error saving system state:', error);
    }
  }

  // Toggle Bot Execution
  startBot() {
    if (this.config.isRunning) return;
    this.config.isRunning = true;
    this.log(`AI Core initialized. Bot STARTED using [${this.config.strategy}] strategy in ${this.config.mode} mode.`);
    this.saveState();

    // Start interval loop (ticks every 3 seconds)
    this.activeIntervalId = setInterval(() => this.tick(), 3000);
    this.triggerUpdate();
  }

  stopBot() {
    if (!this.config.isRunning) return;
    this.config.isRunning = false;
    if (this.activeIntervalId) {
      clearInterval(this.activeIntervalId);
      this.activeIntervalId = null;
    }
    this.log('AI Core suspended. Bot STOPPED.');
    this.saveState();
    this.triggerUpdate();
  }

  // Emergency Liquidation: Close all open positions immediately
  async emergencyCloseAll() {
    this.log('🚨 EMERGENCY LIQUIDATION TRIGGERED: Closing all active vectors...');
    const openTrades = this.trades.filter(t => t.status === 'OPEN');

    for (const trade of openTrades) {
      try {
        const ticker = await this.binanceClient?.getTickerPrice(trade.symbol) || trade.price;
        await this.executeExit(trade, ticker, 'EMERGENCY MANUAL LIQUIDATION');
      } catch (e: any) {
        this.log(`Failed to liquidate trade ${trade.id}: ${e.message}`);
      }
    }
    this.triggerUpdate();
  }

  updateConfig(newConfig: Partial<BotConfig>) {
    const wasRunning = this.config.isRunning;
    if (wasRunning) this.stopBot();

    this.config = { ...this.config, ...newConfig };
    this.initializeBinance();
    this.initializeTelegram(); // Re-initialize Telegram on config update

    if (newConfig.geminiApiKey !== undefined) {
      this.gemmaService.updateApiKey(newConfig.geminiApiKey);
      this.evolutionEngine.updateApiKey(newConfig.geminiApiKey);
    }

    this.saveState();
    if (wasRunning) this.startBot();
    this.triggerUpdate();
  }

  // The active heart of the bot. Runs every 3 seconds.
  private async tick() {
    try {
      // 0. Circuit Breaker Check
      const today = new Date().toISOString().split('T')[0];
      if (this.stats.lastPnLReset !== today) {
        this.stats.dailyPnL = 0;
        this.stats.lastPnLReset = today;
      }

      // Circuit Breaker: Límite de pérdida diaria
      const maxDailyLoss = this.config.tradeSizeUSDT * 2.5;
      if (this.stats.dailyPnL !== undefined && this.stats.dailyPnL < -maxDailyLoss) {
        this.log(`🚨 CIRCUIT BREAKER: Límite de pérdida diaria alcanzado ($${this.stats.dailyPnL.toFixed(2)}). Deteniendo operaciones por seguridad.`);
        this.stopBot();
        return;
      }

      // Meta de Ganancia Diaria: Asegurar beneficios
      const dailyTarget = this.config.dailyProfitTarget || (this.config.tradeSizeUSDT * 1.5);
      if (this.stats.dailyPnL !== undefined && this.stats.dailyPnL >= dailyTarget) {
        this.log(`💰 TARGET REACHED: Meta de ganancia diaria alcanzada ($${this.stats.dailyPnL.toFixed(2)}). Cerrando sesión por hoy con éxito.`);
        this.stopBot();
        return;
      }

      // 1. Fetch current price
      let currentPrice = 0;
      try {
        const client = new BinanceClient({ apiKey: '', apiSecret: '', isTestnet: false, marketType: 'SPOT' });
        currentPrice = await client.getTickerPrice('DOGEUSDT');
      } catch (e: any) {
        this.log(`Binance public feed unavailable: ${e.message}. Using synthetic fallback.`);
        currentPrice = this.generateNextSimulatedPrice();
      }

      // 2. Feed current price into candlestick generator
      this.updateCandles(currentPrice);
      this.pricesBuffer.push(currentPrice);
      if (this.pricesBuffer.length > 1000) this.pricesBuffer.shift(); // Aumentamos el buffer para análisis MTF

      // 3. Compute indicators
      const indicators = calculateIndicators(this.pricesBuffer, this.candles);

      // 4. Update current open trades valuation / PnL & Check trailing Stops / Target Exits
      this.manageOpenPositions(currentPrice);

      // 5. Generate trading signals based on selected strategy
      if (this.config.isRunning) {
        await this.evaluateStrategy(indicators);

        // 5.1 Increment tickCount and trigger periodic mathematical evolution
        this.tickCount++;
        if (this.tickCount % 50 === 0) {
          this.log(`AI Evolutionary cycle triggered. Mutating and backtesting quantitative timing formulas...`);
          this.evolutionEngine.evolve(this.pricesBuffer, this.trades).then((didEvolve) => {
            if (didEvolve) {
              this.log(`🧬 SUCCESS: AI Evolved a superior mathematical formula: ${this.evolutionEngine.getStats().bestFormulaExpression}`);
              this.triggerUpdate();
            }
          }).catch((err) => {
            this.log(`AI Evolution cycle failed: ${err.message}`);
          });
        }

        // 5.2 Trigger periodic Telegram summary
        await this.sendPeriodicSummary();
      }

      this.triggerUpdate();
    } catch (e: any) {
      this.log(`Error inside main tick: ${e.message}`);
    }
  }

  private generateNextSimulatedPrice(): number {
    const lastPrice = this.candles[this.candles.length - 1]?.close || 0.42;
    // Generate beautiful random walk with slight upward drift, fitting DOGE volatile nature
    const noise = (Math.random() - 0.485) * 0.004; // slight uptrend bias
    const wave = Math.sin(Date.now() / 60000) * 0.0008; // smooth cyclic wave
    const change = (noise + wave) * lastPrice;
    return parseFloat((lastPrice + change).toFixed(5));
  }

  private updateCandles(price: number) {
    const now = Math.floor(Date.now() / 1000);
    const lastCandle = this.candles[this.candles.length - 1];

    // Check if 1 minute has elapsed to spawn a new candle
    if (lastCandle && now - lastCandle.time < 60) {
      // Update existing candle
      lastCandle.close = price;
      lastCandle.high = Math.max(lastCandle.high, price);
      lastCandle.low = Math.min(lastCandle.low, price);
      lastCandle.volume += Math.random() * 4000;
    } else {
      // Push new candle
      this.candles.push({
        time: Math.floor(now / 60) * 60,
        open: price,
        high: price,
        low: price,
        close: price,
        volume: Math.random() * 10000 + 5000,
      });
      if (this.candles.length > 200) this.candles.shift();
    }
  }

  private manageOpenPositions(currentPrice: number) {
    const openTrades = this.trades.filter((t) => t.status === 'OPEN');
    if (openTrades.length === 0) return;

    for (const trade of openTrades) {
      const pnl = (currentPrice - trade.price) * trade.quantity * (trade.side === 'BUY' ? 1 : -1);
      const pnlPercent = ((currentPrice - trade.price) / trade.price) * 100 * (trade.side === 'BUY' ? 1 : -1);

      // Actualizar el precio máximo/mínimo alcanzado para el Trailing Stop
      if (trade.side === 'BUY') {
        trade.highestPrice = Math.max(trade.highestPrice || trade.price, currentPrice);
      } else {
        trade.lowestPrice = Math.min(trade.lowestPrice || trade.price, currentPrice);
      }

      trade.pnl = parseFloat(pnl.toFixed(4));
      trade.pnlPercent = parseFloat(pnlPercent.toFixed(2));

      // Lógica de Breakeven: Si ganamos > 1.2%, protegemos la entrada
      if (!trade.isBreakevenActive && pnlPercent >= 1.2) {
        trade.isBreakevenActive = true;
        trade.targetSL = -0.1; // Ponemos el SL un 0.1% arriba/abajo para cubrir fees
        this.log(`🛡️ BREAKEVEN: Operación ${trade.id} protegida. SL movido al precio de entrada.`);
      }

      // Lógica de Trailing Stop: Si el precio cae un X% desde su máximo alcanzado
      const trailingDistance = Math.max(0.8, (trade.targetSL || 2.0) * 0.5);
      const isTrailingStopBreached = trade.side === 'BUY'
        ? (trade.highestPrice && currentPrice < trade.highestPrice * (1 - trailingDistance / 100))
        : (trade.lowestPrice && currentPrice > trade.lowestPrice * (1 + trailingDistance / 100));

      const isStopLossBreached = (trade.targetSL !== undefined && !trade.isBreakevenActive && pnlPercent <= -trade.targetSL) ||
        (trade.isBreakevenActive && pnlPercent <= trade.targetSL!) ||
        isTrailingStopBreached;
      const isTakeProfitBreached = trade.targetTP !== undefined && pnlPercent >= trade.targetTP;

      if (isStopLossBreached || isTakeProfitBreached) {
        let reason = "";
        if (isTakeProfitBreached) reason = `TAKE PROFIT alcanzado (${pnlPercent.toFixed(2)}%)`;
        else if (isTrailingStopBreached) reason = `TRAILING STOP activado (Protegiendo ganancias en ${pnlPercent.toFixed(2)}%)`;
        else reason = `STOP LOSS fijo alcanzado (${pnlPercent.toFixed(2)}%)`;

        this.log(`Automated execution: ${reason}`);
        this.executeExit(trade, currentPrice, reason);

        // Send Telegram notification for automated exit
        // const telegramMsg = `*DOGE Bot Notification* 🤖\n\n` +
        //   `*Trade ID:* ${trade.id}\n` +
        //   `*Action:* ${trade.side === 'BUY' ? 'SELL' : 'BUY'} (Exit)\n` +
        //   `*Exit Price:* $${currentPrice.toFixed(5)}\n` +
        //   `*PnL:* $${pnl.toFixed(2)} (${pnlPercent.toFixed(2)}%)\n` +
        //   `*Reason:* ${reason}`;
        // this.sendTelegramMessage(telegramMsg);
      }
    }
  }

  private async evaluateStrategy(indicators: TechnicalIndicators) {
    // Always trigger background Gemma 4 updates to ensure fresh qualitative/sentiment data is available

    // 3. MEJORA: BTC Correlation Filter
    // Obtenemos el precio de BTC para ver si hay un desplome sistémico
    try {
      const btcPrice = await this.binanceClient?.getTickerPrice('BTCUSDT');
      if (btcPrice) {
        const btcIndicators = calculateIndicators(this.pricesBuffer.map(p => p * (btcPrice / indicators.currentPrice))); // Simulado o real
        if (indicators.currentPrice > indicators.vwap && btcPrice < btcPrice * 0.995) {
          // Si BTC cayó > 0.5% recientemente, precaución extrema
          // this.log('⚠️ Market Alert: BTC momentum is negative. Throttling long entries.');
        }
      }
    } catch (e) { }

    this.triggerBackgroundGemmaFetch(indicators);
    this.triggerBackgroundHmmFetch();

    const openTrades = this.trades.filter((t) => t.status === 'OPEN');
    const hasActiveTrade = openTrades.length > 0;
    const averageEntryPrice = hasActiveTrade
      ? openTrades.reduce((sum, t) => sum + t.price, 0) / openTrades.length
      : 0;

    // Compute real trade statistics for Kelly Criterion
    const closedTrades = this.trades.filter((t) => t.status === 'CLOSED' && t.pnl !== undefined);
    const winningTrades = closedTrades.filter((t) => (t.pnl || 0) > 0);
    const losingTrades = closedTrades.filter((t) => (t.pnl || 0) <= 0);
    const winRate = closedTrades.length > 0 ? winningTrades.length / closedTrades.length : 0.5;
    const avgWin = winningTrades.length > 0
      ? winningTrades.reduce((s, t) => s + (t.pnlPercent || 0), 0) / winningTrades.length / 100
      : 0.012;
    const avgLoss = losingTrades.length > 0
      ? Math.abs(losingTrades.reduce((s, t) => s + (t.pnlPercent || 0), 0) / losingTrades.length) / 100
      : 0.008;
    const tradeStats = { winRate, avgWin, avgLoss };

    // Get dynamically evolved mathematical genes
    const genes = this.evolutionEngine.getActiveGenes();

    let signal: StrategySignal = await this.getGemmaSignal(indicators);

    // Call unified strategy directly instead of just Gemma
    const unifiedSignal = this.strategyManager.getUnifiedSignal(
      indicators,
      this.pricesBuffer,
      hasActiveTrade,
      averageEntryPrice,
      tradeStats,
      genes,
      signal, // pass Gemma as an extra vote
      this.currentRegime?.current_regime,
      this.cachedOrderBook
    );

    // Override local signal variable with the unified signal decision
    signal = unifiedSignal;

    // Process Signal Action
    if (signal.action === 'BUY' && !hasActiveTrade) {
      // MEJORA: Spread Filter
      // Evita entrar si la diferencia entre compra y venta es > 0.15% (baja liquidez)
      try {
        const depth = await this.binanceClient?.getOrderBook('DOGEUSDT');
        if (depth && depth.bids.length > 0 && depth.asks.length > 0) {
          const bestBid = parseFloat(depth.bids[0][0]);
          const bestAsk = parseFloat(depth.asks[0][0]);
          const spread = ((bestAsk - bestBid) / bestBid) * 100;

          if (spread > 0.15) {
            this.log(`BUY Signal ignored: Spread is too wide (${spread.toFixed(3)}%). Liquidity is too low for safe entry.`);
            return;
          }
        }
      } catch (e) { /* Fallback if depth fails */ }

      // MEJORA: Solo comprar si el precio está cerca o por debajo del VWAP (Precio justo)
      // Esto evita comprar en el "pico" de una pompa (FOMO).
      const isFairPrice = indicators.currentPrice <= indicators.vwap * 1.005;

      if (isFairPrice) {
        this.log(`AI Unified strategy generated BUY signal! Reason: ${signal.reason}`);

        // MEJORA: Si la señal no trae Stop Loss, usamos el ATR para un stop de volatilidad
        // Un stop basado en 2.5 * ATR es un estándar profesional para evitar "wicks".
        let dynamicSL = signal.targetSL;
        if (!dynamicSL && indicators.atr > 0) {
          dynamicSL = (indicators.atr * 2.5 / indicators.currentPrice) * 100;
        }

        await this.executeEntry('BUY', indicators.currentPrice, signal.reason, dynamicSL, signal.targetTP);
      } else {
        this.log(`BUY Signal ignored: Price ($${indicators.currentPrice.toFixed(4)}) is too far above VWAP ($${indicators.vwap.toFixed(4)}). Waiting for mean reversion.`);
      }

      // Send Telegram notification for entry
      // const telegramMsg = `*DOGE Bot Notification* 🤖\n\n` +
      //   `*Trade ID:* ${this.trades[this.trades.length - 1]?.id || 'N/A'}\n` +
      //   `*Action:* BUY (Entry)\n` +
      //   `*Entry Price:* $${indicators.currentPrice.toFixed(5)}\n` +
      //   `*Reason:* ${signal.reason}`;
      // this.sendTelegramMessage(telegramMsg);
    } else if (signal.action === 'SELL' && hasActiveTrade) {
      this.log(`AI Unified strategy generated SELL signal! Reason: ${signal.reason}`);
      for (const openTrade of openTrades) {
        await this.executeExit(openTrade, indicators.currentPrice, signal.reason);
      }
    } else if (signal.action === 'BUY' && hasActiveTrade) {
      // Allow safety orders/DCA buy signals if the unified signal suggests it (e.g. from voters)
      this.log(`AI Unified strategy triggered DCA buy! Reason: ${signal.reason}`);
      await this.executeEntry('BUY', indicators.currentPrice, `DCA Safety Order: ${signal.reason}`, signal.targetSL, signal.targetTP);

      // Send Telegram notification for DCA entry
      // const telegramMsg = `*DOGE Bot Notification* 🤖\n\n` +
      //   `*Trade ID:* ${this.trades[this.trades.length - 1]?.id || 'N/A'}\n` +
      //   `*Action:* BUY (DCA Entry)\n` +
      //   `*Entry Price:* $${indicators.currentPrice.toFixed(5)}\n` +
      //   `*Reason:* DCA Safety Order: ${signal.reason}`;
      // this.sendTelegramMessage(telegramMsg);
    }
  }

  // Trigger background Gemma 4 analysis fetch if cache is expired
  private triggerBackgroundGemmaFetch(indicators: TechnicalIndicators) {
    const CACHE_TTL_MS = 10 * 60 * 1000; // 10 minutes
    const now = Date.now();

    // Trigger background refresh if cache is stale and not already fetching
    if (!this.isGemmaFetching && now - this.lastGemmaFetchTime > CACHE_TTL_MS) {
      this.isGemmaFetching = true;
      this.log('🤖 [Gemma 4] Querying Gemma 4 with Google Search Grounding for latest Dogecoin & Bitcoin news (analyzing correlation & reliable sources)...');

      const emaRatio = indicators.ema.ema20 / (indicators.ema.ema50 || 1.0);
      const bbRange = indicators.bollinger.upper - indicators.bollinger.lower || 0.0001;
      const bbPosition = (indicators.currentPrice - indicators.bollinger.lower) / bbRange;

      this.gemmaService.getSignal(
        'DOGE/USDT',
        indicators.currentPrice,
        { rsi: indicators.rsi, macdHist: indicators.macd.hist, emaRatio, bbPosition },
        this.candles.slice(-15)
      ).then((gemmaSignal) => {
        this.cachedGemmaSignal = gemmaSignal;
        this.lastGemmaFetchTime = Date.now();
        this.isGemmaFetching = false;
        this.log(`🤖 [Gemma 4] Signal updated → ${gemmaSignal.action} (Confidence: ${(gemmaSignal.confidence * 100).toFixed(0)}%) | ${gemmaSignal.reason.substring(0, 200)}...`);
        this.triggerUpdate();
      }).catch((err) => {
        this.isGemmaFetching = false;
        this.log(`🤖 [Gemma 4] Signal fetch failed: ${err.message}`);
      });
    }
  }

  // Trigger background HMM analysis
  private triggerBackgroundHmmFetch() {
    const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes
    const now = Date.now();

    if (!this.isHmmFetching && now - this.lastHmmFetchTime > CACHE_TTL_MS) {
      this.isHmmFetching = true;
      this.log('🤖 [HMM] Analyzing mathematical market regimes using Hidden Markov Models...');

      const volumes = this.candles.map(c => c.volume);

      hmmService.getCurrentRegime(this.pricesBuffer, volumes).then((regime) => {
        this.currentRegime = regime;
        this.lastHmmFetchTime = Date.now();
        this.isHmmFetching = false;
        this.log(`🤖 [HMM] Market Regime Detected: ${regime.current_regime}`);
        this.triggerUpdate();
      }).catch((err) => {
        this.isHmmFetching = false;
        this.log(`🤖 [HMM] Failed to detect regime: ${err.message}`);
      });
    }
  }

  // Gemma 4 strategy: query the LLM every 10 minutes; use cached result between calls
  private async getGemmaSignal(indicators: TechnicalIndicators): Promise<StrategySignal> {
    this.triggerBackgroundGemmaFetch(indicators);

    // Return cached signal or HOLD if we haven't received one yet
    if (this.cachedGemmaSignal) {
      const now = Date.now();
      const ageMinutes = ((now - this.lastGemmaFetchTime) / 60000).toFixed(1);
      return {
        action: this.cachedGemmaSignal.action,
        confidence: this.cachedGemmaSignal.confidence,
        targetSL: this.cachedGemmaSignal.stopLossPercent,
        targetTP: this.cachedGemmaSignal.takeProfitPercent,
        reason: `🤖 AI Unified (${ageMinutes}min ago): ${this.cachedGemmaSignal.reason}`,
      };
    }

    return {
      action: 'HOLD',
      confidence: 0.5,
      reason: '🤖 AI Unified: Awaiting first analysis from Gemini + Google Search...',
    };
  }

  private async executeEntry(side: 'BUY' | 'SELL', price: number, reason: string, targetSL?: number, targetTP?: number) {
    const symbol = 'DOGEUSDT';

    // 1. MEJORA: Risk-Based Position Sizing (ATR + SL)
    // Calculamos el tamaño basado en cuánto dinero estamos dispuestos a perder si toca el SL.
    const balance = this.stats.totalBalanceUSDT;
    const riskAmount = balance * (MAX_RISK_PER_TRADE_PERCENT / 100);

    // Si no hay SL definido, usamos un default de 2% para el cálculo de riesgo
    const slDistancePercent = targetSL || 2.0;

    // Cantidad basada en Riesgo: riskAmount / (distancia al SL en precio)
    const riskBasedQty = riskAmount / (price * (slDistancePercent / 100));

    // 2. Kelly Sizing dinámico (como multiplicador de confianza)
    const kellyFrac = this.evolutionEngine.getActiveGenes().kellyFraction || 0.2;
    const kellyBasedQty = (balance * kellyFrac) / price;

    // Tomamos la más conservadora de las dos para evitar sobre-apalancamiento
    let quantity = Math.min(riskBasedQty, kellyBasedQty);

    // Límite de seguridad: No exceder el tradeSizeUSDT base configurado por el usuario
    // a menos que la confianza sea extrema.
    const maxQuantityFromConfig = (this.config.tradeSizeUSDT * 2) / price;
    quantity = Math.min(quantity, maxQuantityFromConfig);

    // Asegurar cantidad mínima para Binance
    quantity = Math.max(quantity, 100); // Mínimo ~4-10 USDT en DOGE

    const amount = quantity * price;

    this.log(`Initiating position entry vector... ${side} ${quantity.toFixed(1)} DOGE @ $${price.toFixed(5)} (~$${amount} USDT)`);

    if (this.binanceClient) {
      try {
        const isFutures = this.config.marketType === 'FUTURES';
        // En Hedge Mode, necesitamos especificar positionSide: LONG o SHORT
        const positionSide = isFutures ? (side === 'BUY' ? 'LONG' : 'SHORT') : undefined;
        const order = await this.binanceClient.placeOrder(symbol, side, 'MARKET', quantity, undefined, false, positionSide);
        const fillPrice = (order.avgPrice && parseFloat(order.avgPrice) > 0)
          ? parseFloat(order.avgPrice)
          : (order.fills && order.fills.length > 0
            ? parseFloat(order.fills[0].price)
            : price);
        const fillQty = (order.executedQty && parseFloat(order.executedQty) > 0)
          ? parseFloat(order.executedQty)
          : (order.origQty && parseFloat(order.origQty) > 0
            ? parseFloat(order.origQty)
            : quantity);

        const trade: Trade = {
          id: order.orderId.toString(),
          symbol,
          side,
          type: this.config.mode,
          price: fillPrice,
          quantity: fillQty,
          amount: fillPrice * fillQty,
          timestamp: Date.now(),
          status: 'OPEN',
          reason,
          targetSL,
          targetTP,
          highestPrice: fillPrice,
          lowestPrice: fillPrice,
        };

        this.trades.push(trade);
        this.log(`Binance Order filled successfully. ID: ${trade.id}. Price: $${fillPrice.toFixed(5)}`);

        // Sync real-time balances if needed
        await this.syncRealAccountBalances();
      } catch (e: any) {
        this.log(`ERROR: Binance failed to execute order: ${e.message}`);
      }
    } else {
      this.log(`Execution halted: Binance client not initialized.`);
    }
  }

  private async executeExit(trade: Trade, price: number, reason: string) {
    this.log(`Closing position vector ${trade.id} @ $${price.toFixed(5)}... Reason: ${reason}`);

    if (this.binanceClient) {
      try {
        const exitSide = trade.side === 'BUY' ? 'SELL' : 'BUY';
        // En FUTURES usamos reduceOnly: true para asegurar que solo cerramos el volumen actual
        const isFutures = this.config.marketType === 'FUTURES';
        // En Hedge Mode, positionSide debe coincidir con el lado de la posición original
        const positionSide = isFutures ? (trade.side === 'BUY' ? 'LONG' : 'SHORT') : undefined;
        const order = await this.binanceClient.placeOrder(trade.symbol, exitSide, 'MARKET', trade.quantity, undefined, true, positionSide);

        const fillPrice = (order.avgPrice && parseFloat(order.avgPrice) > 0)
          ? parseFloat(order.avgPrice)
          : (order.fills && order.fills.length > 0
            ? parseFloat(order.fills[0].price)
            : price);

        // Finalize Trade structure
        trade.status = 'CLOSED';
        trade.exitPrice = fillPrice;
        trade.exitTimestamp = Date.now();

        const pnl = (fillPrice - trade.price) * trade.quantity * (trade.side === 'BUY' ? 1 : -1);
        trade.pnl = parseFloat(pnl.toFixed(4));
        trade.pnlPercent = parseFloat((((fillPrice - trade.price) / trade.price) * 100 * (trade.side === 'BUY' ? 1 : -1)).toFixed(2));

        // Update Daily PnL
        this.stats.dailyPnL = (this.stats.dailyPnL || 0) + pnl;

        this.log(`Binance exit order filled. PnL: $${trade.pnl.toFixed(2)} (${trade.pnlPercent.toFixed(2)}%)`);

        // Enviar notificación de Telegram para salida REAL
        // const telegramMsg = `*DOGE Bot Notification (REAL)* 🤖\n\n` +
        //   `*Trade ID:* ${trade.id}\n` +
        //   `*Action:* ${trade.side === 'BUY' ? 'SELL' : 'BUY'} (Exit)\n` +
        //   `*Exit Price:* $${fillPrice.toFixed(5)}\n` +
        //   `*PnL:* $${trade.pnl.toFixed(2)} (${trade.pnlPercent.toFixed(2)}%)\n` +
        //   `*Reason:* ${reason}`;
        //
        // No bloqueamos el hilo principal, enviamos en segundo plano
        // this.sendTelegramMessage(telegramMsg);

        // Train Neural Network using trade feedback!
        this.trainModelOnExit(trade);

        this.updateStats();
        await this.syncRealAccountBalances();
      } catch (e: any) {
        this.log(`ERROR: Binance failed to close position: ${e.message}`);
      }
    } else {
      this.log(`Exit halted: Binance client not initialized.`);
    }
  }

  // Active reinforcement learning call
  private trainModelOnExit(trade: Trade) {
    if (trade.pnl === undefined) return;

    // Fetch last candle features
    const rsi = this.candles[this.candles.length - 1]?.close || 50; // indicator proxy
    const indicators = calculateIndicators(this.pricesBuffer);

    const emaRatio = indicators.ema.ema20 / (indicators.ema.ema50 || 1.0);
    const bbRange = indicators.bollinger.upper - indicators.bollinger.lower || 0.0001;
    const bbPosition = (trade.price - indicators.bollinger.lower) / bbRange;

    const normalizedInputs = CustomNeuralNetwork.normalizeIndicators(
      indicators.rsi,
      indicators.macd.hist,
      emaRatio,
      bbPosition,
      indicators.botActivity
    );

    // Reinforce the neural network weights!
    this.neuralNet.reinforce(normalizedInputs, trade.side, trade.pnl);
    this.log(`Neural Core weights reinforced. Backpropagation feedback applied successfully.`);

    // If trade resulted in a loss, trigger Genetic Algorithm mathematical evolution!
    if (trade.pnl <= 0) {
      this.log(`Underperforming closed position [${trade.id}] detected. Initiating Genetic Mathematical Evolution Epoch...`);
      this.evolutionEngine.evolve(this.pricesBuffer, this.trades).then((didEvolve) => {
        if (didEvolve) {
          this.log(`🧬 EVOLVED: AI evolved a superior chromosome. New formula: ${this.evolutionEngine.getStats().bestFormulaExpression}`);
          this.triggerUpdate();
        }
      }).catch((err) => {
        this.log(`AI Evolution cycle failed: ${err.message}`);
      });
    }
  }

  // Update statistics calculations
  private updateStats() {
    const closed = this.trades.filter((t) => t.status === 'CLOSED');
    if (closed.length === 0) return;

    const netProfit = closed.reduce((sum, t) => sum + (t.pnl || 0), 0);
    const winning = closed.filter((t) => (t.pnl || 0) > 0);
    const losing = closed.filter((t) => (t.pnl || 0) <= 0);

    const winRate = (winning.length / closed.length) * 100;

    let totalGains = winning.reduce((sum, t) => sum + (t.pnl || 0), 0);
    let totalLosses = Math.abs(losing.reduce((sum, t) => sum + (t.pnl || 0), 0));
    const profitFactor = totalLosses === 0 ? totalGains : totalGains / totalLosses;

    // In simulated demo, net profit scales portfolio. In Real/Testnet it mirrors trades.
    this.stats.netProfitUSDT = parseFloat(netProfit.toFixed(2));
    this.stats.winRatePercent = parseFloat(winRate.toFixed(1));
    this.stats.profitFactor = parseFloat(profitFactor.toFixed(2));
    this.stats.totalTrades = closed.length;
    this.stats.winningTrades = winning.length;
    this.stats.losingTrades = losing.length;
  }

  // Sync balances with real Binance API
  private async syncRealAccountBalances() {
    if (!this.binanceClient) return;

    try {
      const accountInfo = await this.binanceClient.getAccountInfo();

      if (this.config.marketType === 'FUTURES') {
        const assets = accountInfo.assets || [];
        const positions = accountInfo.positions || [];

        const usdtAsset = assets.find((a: any) => a.asset === 'USDT');
        const usdtFree = usdtAsset ? parseFloat(usdtAsset.availableBalance) : 0;

        const dogePosition = positions.find((p: any) => p.symbol === 'DOGEUSDT');
        const dogeAmt = dogePosition ? parseFloat(dogePosition.positionAmt) : 0;

        this.stats.totalBalanceUSDT = parseFloat(usdtFree.toFixed(2));
        this.stats.dogeBalance = parseFloat(dogeAmt.toFixed(2));

        this.log(`Synced Futures balances from Binance: ${this.stats.totalBalanceUSDT} USDT (Available Margin), ${this.stats.dogeBalance} DOGE (Position).`);
      } else {
        const balances = accountInfo.balances || [];

        const usdtAsset = balances.find((b: any) => b.asset === 'USDT');
        const dogeAsset = balances.find((b: any) => b.asset === 'DOGE');

        const usdtFree = usdtAsset ? parseFloat(usdtAsset.free) : 0;
        const dogeFree = dogeAsset ? parseFloat(dogeAsset.free) : 0;

        this.stats.totalBalanceUSDT = parseFloat(usdtFree.toFixed(2));
        this.stats.dogeBalance = parseFloat(dogeFree.toFixed(2));

        this.log(`Synced Spot balances from Binance: ${this.stats.totalBalanceUSDT} USDT, ${this.stats.dogeBalance} DOGE.`);
      }
    } catch (e: any) {
      this.log(`Error syncing balances: ${e.message}`);
    }
  }

  // Package state for websocket JSON payload
  getStatePayload() {
    const currentPrice = this.candles[this.candles.length - 1]?.close || 0;
    const indicators = calculateIndicators(this.pricesBuffer);

    return {
      config: this.config,
      stats: this.stats,
      trades: this.trades,
      logs: this.getLogs(),
      candles: this.candles.slice(-80), // keep only what's needed for standard graph
      indicators: {
        rsi: indicators.rsi,
        macd: indicators.macd,
        ema: indicators.ema,
        bollinger: indicators.bollinger,
        currentPrice,
      },
      hmmRegime: this.currentRegime,
      neuralNetwork: this.neuralNet.getWeightsAndNeurons(),
      evolution: this.evolutionEngine.getStats(),
    };
  }

  public sendTelegramTest(token: string, chatId: string) {
    this.log(`Enviando mensaje de prueba de Telegram con Token: ${token.substring(0, 10)}... y Chat ID: ${chatId}`);
    const saved = { token: this.config.telegramBotToken, chatId: this.config.telegramChatId };
    this.config.telegramBotToken = token;
    this.config.telegramChatId = chatId;
    this.sendTelegramMessage('🤖 *Test de Telegram exitoso* — Bot DOGE/USDT conectado correctamente.', true)
      .then(() => {
        this.log(`Test de Telegram completado. Guardando credenciales.`);
        this.saveState();
      })
      .catch((err) => {
        this.log(`Error al enviar mensaje de prueba: ${err.message}`);
        this.config.telegramBotToken = saved.token;
        this.config.telegramChatId = saved.chatId;
      });
  }

  private async sendTelegramMessage(text: string, throwError: boolean = false) {
    const token = this.config.telegramBotToken;
    const chatId = this.config.telegramChatId;

    if (!token || !chatId) return;

    try {
      const response = await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chat_id: chatId, text: text, parse_mode: 'Markdown' })
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Telegram API responded with status ${response.status}: ${errorText}`);
      }
    } catch (error: any) {
      this.log(`Telegram notification error: ${error.message}`);
      if (throwError) throw error;
    }
  }

  /** Fuerza el envío inmediato del reporte de Telegram y reinicia el contador de 3h */
  public async sendSummaryNow() {
    this.lastSummaryTime = 0; // Reset so sendPeriodicSummary fires right away
    await this.sendPeriodicSummary();
  }

  /** Genera un reporte detallado de la situación actual del bot para Telegram */
  private async generateStatusReport(): Promise<string> {
    const openTrades = this.trades.filter(t => t.status === 'OPEN');
    const closedTrades = this.trades.filter(t => t.status === 'CLOSED');
    const netProfit = this.stats.netProfitUSDT;

    // ROI Basado en el balance inicial de 10,000 (valor inicial de las stats)
    const roi = (netProfit / 10000) * 100;
    const runningEmoji = this.config.isRunning ? '🚀 *EJECUTANDO*' : '💤 *PAUSADO*';
    const marketNews = await this.gemmaService.generateMarketSummary();

    let openSummary = '';
    if (openTrades.length === 0) {
      openSummary = '_Sin posiciones activas._';
    } else {
      openSummary = openTrades.map(t => {
        const pnlStr = (t.pnlPercent || 0) >= 0 ? `+${(t.pnlPercent || 0).toFixed(2)}%` : `${(t.pnlPercent || 0).toFixed(2)}%`;
        const emoji = (t.pnlPercent || 0) >= 0 ? '🟢' : '🔴';
        return `• ${t.side} | $${t.amount.toFixed(2)} | ${emoji} ${pnlStr}`;
      }).join('\n');
    }

    return `📊 *REPORTE DE ESTADO* \n` +
      `Motor: ${runningEmoji}\n` +
      `ROI Total: *${roi >= 0 ? '📈' : '📉'} ${roi.toFixed(2)}%*\n\n` +
      `*📦 Operaciones Abiertas:* \n${openSummary}\n\n` +
      `*✅ Operaciones Cerradas:* ${closedTrades.length} (Win Rate: ${this.stats.winRatePercent}%)\n\n` +
      `*🤖 Visión de la IA (Gemma 4):*\n${marketNews}`;
  }

  private async sendPeriodicSummary() {
    const SUMMARY_INTERVAL_MS = 3 * 60 * 60 * 1000; // 3 hours
    const now = Date.now();

    if (now - this.lastSummaryTime < SUMMARY_INTERVAL_MS) {
      return; // Not time yet
    }

    if (!this.config.telegramBotToken || !this.config.telegramChatId) {
      return; // Telegram not configured
    }

    this.lastSummaryTime = now;
    this.log('📊 Generando reporte periódico de 3 horas para Telegram...');

    // 1. Open Trades
    const openTrades = this.trades.filter(t => t.status === 'OPEN');
    let openSummary = '';
    if (openTrades.length === 0) {
      openSummary = 'Ninguna operación abierta.';
    } else {
      openSummary = openTrades.map(t => {
        const pnlStr = (t.pnlPercent || 0) >= 0 ? `+${(t.pnlPercent || 0).toFixed(2)}%` : `${(t.pnlPercent || 0).toFixed(2)}%`;
        const emoji = (t.pnlPercent || 0) >= 0 ? '🟢' : '🔴';
        return `• ${t.side} $${t.amount.toFixed(2)} | PnL: ${emoji} ${pnlStr}`;
      }).join('\n');
    }

    // 2. Closed Trades in the last 3 hours
    const threeHoursAgo = now - SUMMARY_INTERVAL_MS;
    const recentClosed = this.trades.filter(t => t.status === 'CLOSED' && (t.exitTimestamp || 0) > threeHoursAgo);
    let closedSummary = '';
    let recentNetProfit = 0;
    if (recentClosed.length === 0) {
      closedSummary = 'Ninguna operación cerrada en las últimas 3 horas.';
    } else {
      recentNetProfit = recentClosed.reduce((sum, t) => sum + (t.pnl || 0), 0);
      const winCount = recentClosed.filter(t => (t.pnl || 0) > 0).length;
      const lossCount = recentClosed.length - winCount;
      const emoji = recentNetProfit >= 0 ? '🤑' : '📉';
      closedSummary = `Se cerraron ${recentClosed.length} operaciones (${winCount}W / ${lossCount}L).\n` +
        `PnL del periodo: ${emoji} $${recentNetProfit.toFixed(2)}`;
    }

    // 3. Market Summary from Gemma
    const marketNews = await this.gemmaService.generateMarketSummary();

    // 4. HMM Regime
    const regimeStr = this.currentRegime ? this.currentRegime.current_regime : 'Desconocido';

    // Construct the final message
    const message = `*DOGE Bot | Reporte de 3 Horas* 🕒\n\n` +
      `*1️⃣ Operaciones Abiertas (${openTrades.length})*\n${openSummary}\n\n` +
      `*2️⃣ Actividad Reciente*\n${closedSummary}\n\n` +
      `*3️⃣ Régimen de Mercado (HMM)*\n🧠 Detectado: \`${regimeStr}\`\n\n` +
      `*4️⃣ Visión del Mercado & Noticias*\n${marketNews}`;

    await this.sendTelegramMessage(message);
    this.log('✅ Reporte de 3 horas enviado a Telegram exitosamente.');
  }
}
