import React, { useState, useEffect, useRef } from 'react';
import {
  Play, Square, Bot, TrendingUp, Cpu, Settings, Shield,
  Terminal, Activity, DollarSign, Percent, Award,
  Key, Loader2, Sparkles, Send
} from 'lucide-react';
import confetti from 'canvas-confetti';

interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface Trade {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  type: string;
  price: number;
  quantity: number;
  amount: number;
  timestamp: number;
  status: 'OPEN' | 'CLOSED';
  pnl?: number;
  pnlPercent?: number;
  exitPrice?: number;
  exitTimestamp?: number;
  reason?: string;
}

interface NeuralNetwork {
  inputToHidden: number[][];
  hiddenToOutput: number[][];
  hiddenBiases: number[];
  outputBiases: number[];
  inputs: number[];
  hidden: number[];
  outputs: number[];
}

interface MathGenes {
  binomialThreshold: number;
  zScoreEntry: number;
  zScoreExit: number;
  hurstTrending: number;
  hurstReversion: number;
  kalmanNoiseRatio: number;
  varianceRatioLongPeriod: number;
  kellyFraction: number;
  momentumLookback: number;
}

interface EvolutionStats {
  generation: number;
  activeGenes: MathGenes;
  fitnessScore: number;
  lastEvolvedAt: number;
  bestFormulaExpression: string;
  evolutionLogs: string[];
}

interface BotConfig {
  mode: 'DEMO' | 'TESTNET' | 'REAL';
  isRunning: boolean;
  strategy: 'ORACLE' | 'GRID_DCA' | 'NEURAL_NETWORK' | 'CONSERVATIVE';
  tradeSizeUSDT: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  binanceApiKey: string;
  binanceApiSecret: string;
  gridLayers: number;
  marketType: 'SPOT' | 'FUTURES';
  leverage: number; // Only for FUTURES
  telegramBotToken?: string;
  telegramChatId?: string;
}

interface BotStats {
  totalBalanceUSDT: number;
  dogeBalance: number;
  netProfitUSDT: number;
  winRatePercent: number;
  profitFactor: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
}

interface Indicators {
  rsi: number;
  macd: { macd: number; signal: number; hist: number };
  ema: { ema20: number; ema50: number; ema200: number };
  bollinger: { upper: number; middle: number; lower: number; width: number };
  currentPrice: number;
}

export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [config, setConfig] = useState<BotConfig>({
    mode: 'DEMO',
    isRunning: false,
    strategy: 'ORACLE',
    tradeSizeUSDT: 50,
    stopLossPercent: 2.0,
    takeProfitPercent: 1.5,
    binanceApiKey: '',
    binanceApiSecret: '',
    gridLayers: 3,
    marketType: 'SPOT', // Default to SPOT
    leverage: 5, // Default leverage for futures
    telegramBotToken: '7575795641:AAHdzUClOsiwyqp4mZorLEyvDqeoYIh2LKA',
    telegramChatId: '7543101442'
  });

  const [stats, setStats] = useState<BotStats>({
    totalBalanceUSDT: 10000.0,
    dogeBalance: 0,
    netProfitUSDT: 0,
    winRatePercent: 0,
    profitFactor: 0,
    totalTrades: 0,
    winningTrades: 0,
    losingTrades: 0
  });

  const [trades, setTrades] = useState<Trade[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [indicators, setIndicators] = useState<Indicators>({
    rsi: 50,
    macd: { macd: 0, signal: 0, hist: 0 },
    ema: { ema20: 0.42, ema50: 0.42, ema200: 0.42 },
    bollinger: { upper: 0.43, middle: 0.42, lower: 0.41, width: 0.05 },
    currentPrice: 0.42
  });

  const [neuralNet, setNeuralNet] = useState<NeuralNetwork>({
    inputToHidden: [],
    hiddenToOutput: [],
    hiddenBiases: [],
    outputBiases: [],
    inputs: [0, 0, 0, 0],
    hidden: Array(8).fill(0),
    outputs: [0, 0]
  });

  const [evolution, setEvolution] = useState<EvolutionStats>({
    generation: 0,
    activeGenes: {
      binomialThreshold: 0.70,
      zScoreEntry: -2.0,
      zScoreExit: 1.5,
      hurstTrending: 0.55,
      hurstReversion: 0.45,
      kalmanNoiseRatio: 0.005,
      varianceRatioLongPeriod: 10,
      kellyFraction: 0.25,
      momentumLookback: 20
    },
    fitnessScore: 0.0,
    lastEvolvedAt: Date.now(),
    bestFormulaExpression: "P(X >= k | n, p) * K_vel + Kelly(f*) - Z_score * VarianceRatio()",
    evolutionLogs: ["Genetic engine standby."]
  });

  // Settings modification state
  const [editApiKey, setEditApiKey] = useState('');
  const [editApiSecret, setEditApiSecret] = useState('');
  const [editTradeSize, setEditTradeSize] = useState(50);
  const [editStopLoss, setEditStopLoss] = useState(2.0);
  const [editTakeProfit, setEditTakeProfit] = useState(1.5);
  const [editStrategy, setEditStrategy] = useState<'ORACLE' | 'GRID_DCA' | 'NEURAL_NETWORK' | 'CONSERVATIVE'>('ORACLE');
  const [editMode, setEditMode] = useState<'DEMO' | 'TESTNET' | 'REAL'>('DEMO');
  const [editMarketType, setEditMarketType] = useState<'SPOT' | 'FUTURES'>('SPOT');
  const [editLeverage, setEditLeverage] = useState(5);
  const [editTelegramBotToken, setEditTelegramBotToken] = useState('');
  const [editTelegramChatId, setEditTelegramChatId] = useState('');

  const [priceDirection, setPriceDirection] = useState<'UP' | 'DOWN' | 'STABLE'>('STABLE');
  const lastPriceRef = useRef<number>(0.42);
  const socketRef = useRef<WebSocket | null>(null);
  const [isUpdatingConfig, setIsUpdatingConfig] = useState(false);

  // Success confetti trigger on trade win
  const lastTradeCount = useRef(0);
  useEffect(() => {
    const closedTrades = trades.filter(t => t.status === 'CLOSED');
    if (closedTrades.length > lastTradeCount.current) {
      const latestClosed = closedTrades[closedTrades.length - 1];
      if (latestClosed && latestClosed.pnl && latestClosed.pnl > 0) {
        confetti({
          particleCount: 100,
          spread: 70,
          origin: { y: 0.6 },
          colors: ['#00e5ff', '#00ff88', '#ffb700']
        });
      }
      lastTradeCount.current = closedTrades.length;
    }
  }, [trades]);

  // Connect to backend WS server
  useEffect(() => {
    const connectWS = () => {
      const socket = new WebSocket('ws://localhost:5000');
      socketRef.current = socket;

      socket.onopen = () => {
        setIsConnected(true);
        console.log('Connected to AI Bot backend');
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // Set states
          setConfig(data.config);
          setStats(data.stats);
          setTrades(data.trades);
          setLogs(data.logs);
          setCandles(data.candles);
          setIndicators(data.indicators);

          if (data.neuralNetwork) {
            setNeuralNet(data.neuralNetwork);
          }

          if (data.evolution) {
            setEvolution(data.evolution);
          }

          // Visual price ticker flashing
          const newPrice = data.indicators.currentPrice;
          if (newPrice > lastPriceRef.current) {
            setPriceDirection('UP');
          } else if (newPrice < lastPriceRef.current) {
            setPriceDirection('DOWN');
          }
          lastPriceRef.current = newPrice;
        } catch (error) {
          console.error('Error parsing WS message:', error);
        }
      };

      socket.onclose = () => {
        setIsConnected(false);
        // Retry connection after 3 seconds
        setTimeout(connectWS, 3000);
      };
    };

    connectWS();

    return () => {
      if (socketRef.current) socketRef.current.close();
    };
  }, []);

  // Update visual configuration form fields when server config loads
  useEffect(() => {
    setEditApiKey(config.binanceApiKey || '');
    setEditApiSecret(config.binanceApiSecret || '');
    setEditTradeSize(config.tradeSizeUSDT);
    setEditStopLoss(config.stopLossPercent);
    setEditTakeProfit(config.takeProfitPercent);
    setEditStrategy(config.strategy);
    setEditMode(config.mode);
    setEditMarketType(config.marketType || 'SPOT');
    setEditLeverage(config.leverage || 5);
    setEditTelegramBotToken(config.telegramBotToken || '');
    setEditTelegramChatId(config.telegramChatId || '');
  }, [config]);

  // Trigger server actions
  const toggleBot = async () => {
    const endpoint = config.isRunning ? 'stop' : 'start';
    try {
      await fetch(`http://localhost:5000/api/${endpoint}`, { method: 'POST' });
    } catch (e) {
      console.error(`Error toggling bot to ${endpoint}:`, e);
    }
  };

  const saveConfiguration = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsUpdatingConfig(true);
    try {
      const response = await fetch('http://localhost:5000/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: editMode,
          strategy: editStrategy,
          tradeSizeUSDT: Number(editTradeSize),
          stopLossPercent: Number(editStopLoss),
          takeProfitPercent: Number(editTakeProfit),
          binanceApiKey: editApiKey,
          binanceApiSecret: editApiSecret,
          marketType: editMarketType,
          leverage: Number(editLeverage),
          telegramBotToken: editTelegramBotToken,
          telegramChatId: editTelegramChatId
        })
      });
      if (response.ok) {
        // Trigger a simple notification or success animation
      }
    } catch (e) {
      console.error('Error updating config matrix:', e);
    } finally {
      setIsUpdatingConfig(false);
    }
  };

  // Función para exportar historial a CSV para análisis matemático
  const exportTradeHistory = () => {
    if (trades.length === 0) return;

    const headers = ["ID", "Side", "Entry Price", "Exit Price", "Quantity", "Amount", "PnL USDT", "PnL %", "Reason", "Timestamp"];
    const rows = trades.map(t => [
      t.id,
      t.side,
      t.price,
      t.exitPrice || '',
      t.quantity,
      t.amount,
      t.pnl || '',
      t.pnlPercent || '',
      `"${t.reason || ''}"`,
      new Date(t.timestamp).toISOString()
    ]);

    const csvContent = [headers, ...rows].map(e => e.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", `trade_history_${new Date().getTime()}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Función para enviar un mensaje de prueba a Telegram
  const sendTestTelegramMessage = async () => {
    if (!editTelegramBotToken || !editTelegramChatId) {
      alert('Por favor, introduce el Token del Bot y el Chat ID de Telegram.');
      return;
    }
    await fetch('http://localhost:5000/api/send-telegram-test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: editTelegramBotToken, chatId: editTelegramChatId })
    });
  };

  // Helper formatting values
  const formatCurrency = (val: number, decimals = 2) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    }).format(val);
  };

  // SVG Chart component
  const renderCandlestickChart = () => {
    if (candles.length === 0) {
      return (
        <div className="flex flex-col items-center justify-center h-full text-slate-500 gap-2">
          <Loader2 className="animate-spin text-neon-cyan" size={32} />
          <span>Warm feeding market stream data...</span>
        </div>
      );
    }

    const margin = { top: 20, right: 60, bottom: 30, left: 10 };
    const width = 600;
    const height = 350;
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    // Extrema calculations
    const highPrices = candles.map(c => c.high);
    const lowPrices = candles.map(c => c.low);
    const maxPrice = Math.max(...highPrices, indicators.bollinger.upper) * 1.002;
    const minPrice = Math.min(...lowPrices, indicators.bollinger.lower) * 0.998;
    const priceRange = maxPrice - minPrice;

    // Helper to map values to coordinates
    const getX = (index: number) => margin.left + (index / (candles.length - 1)) * chartWidth;
    const getY = (price: number) => margin.top + chartHeight - ((price - minPrice) / priceRange) * chartHeight;

    // Build Bollinger Band fill path
    let bollingerPath = '';
    if (candles.length > 0) {
      const topPoints = candles.map((_, i) => `${getX(i)},${getY(indicators.bollinger.upper)}`);
      const bottomPoints = candles.map((_, i) => `${getX(i)},${getY(indicators.bollinger.lower)}`).reverse();
      bollingerPath = `M ${topPoints.join(' L ')} L ${bottomPoints.join(' L ')} Z`;
    }

    // Build EMA paths
    const ema20Points = candles.map((_, i) => `${getX(i)},${getY(indicators.ema.ema20)}`).join(' L ');
    const ema50Points = candles.map((_, i) => `${getX(i)},${getY(indicators.ema.ema50)}`).join(' L ');

    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full">
        <defs>
          <linearGradient id="bbGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(0, 229, 255, 0.05)" />
            <stop offset="100%" stopColor="rgba(0, 229, 255, 0.01)" />
          </linearGradient>
          <linearGradient id="gridLines" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="rgba(255,255,255,0.01)" />
            <stop offset="100%" stopColor="rgba(255,255,255,0.03)" />
          </linearGradient>
        </defs>

        {/* Horizontal gridlines */}
        {Array.from({ length: 5 }).map((_, i) => {
          const p = minPrice + (priceRange / 4) * i;
          const y = getY(p);
          return (
            <g key={i}>
              <line x1={margin.left} y1={y} x2={width - margin.right} y2={y} stroke="rgba(255,255,255,0.05)" strokeDasharray="3,3" />
              <text x={width - margin.right + 8} y={y + 4} fill="rgba(255,255,255,0.4)" fontSize="9" fontFamily="monospace">
                ${p.toFixed(4)}
              </text>
            </g>
          );
        })}

        {/* Bollinger Band cloud */}
        {bollingerPath && (
          <path d={bollingerPath} fill="url(#bbGradient)" stroke="rgba(0, 229, 255, 0.15)" strokeWidth="0.8" />
        )}

        {/* EMA curves */}
        {ema50Points && <path d={ema50Points} fill="none" stroke="#ff007f" strokeWidth="1.2" opacity="0.6" strokeDasharray="2,2" />}
        {ema20Points && <path d={ema20Points} fill="none" stroke="#00e5ff" strokeWidth="1.5" opacity="0.8" />}

        {/* Candlesticks */}
        {candles.map((candle, idx) => {
          const x = getX(idx);
          const yOpen = getY(candle.open);
          const yClose = getY(candle.close);
          const yHigh = getY(candle.high);
          const yLow = getY(candle.low);

          const isBullish = candle.close >= candle.open;
          const bodyColor = isBullish ? '#00ff88' : '#ff3366';
          const bodyWidth = Math.max(2, chartWidth / candles.length - 2);

          return (
            <g key={idx} opacity={idx === candles.length - 1 ? 1 : 0.85}>
              {/* Wick */}
              <line x1={x} y1={yHigh} x2={x} y2={yLow} stroke={bodyColor} strokeWidth="1" />
              {/* Body */}
              <rect
                x={x - bodyWidth / 2}
                y={Math.min(yOpen, yClose)}
                width={bodyWidth}
                height={Math.max(1.5, Math.abs(yOpen - yClose))}
                fill={bodyColor}
                rx="0.5"
              />
            </g>
          );
        })}

        {/* Trade execution markers */}
        {trades.map((trade, idx) => {
          const candleIdx = candles.findIndex(c => Math.abs(c.time - Math.floor(trade.timestamp / 60000) * 60) < 60);
          if (candleIdx === -1) return null;

          const cx = getX(candleIdx);
          const cy = getY(trade.price);
          const isBuy = trade.side === 'BUY';

          return (
            <g key={idx}>
              <circle cx={cx} cy={cy} r="6" fill={isBuy ? '#00ff88' : '#ff3366'} stroke="#fff" strokeWidth="1" />
              <path
                d={isBuy ? `M ${cx} ${cy - 12} L ${cx - 4} ${cy - 8} L ${cx + 4} ${cy - 8} Z` : `M ${cx} ${cy + 12} L ${cx - 4} ${cy + 8} L ${cx + 4} ${cy + 8} Z`}
                fill={isBuy ? '#00ff88' : '#ff3366'}
              />
            </g>
          );
        })}
      </svg>
    );
  };

  // SVG Neural Network Visualizer component
  const renderNeuralNetwork = () => {
    const { inputToHidden, hiddenToOutput, inputs, hidden, outputs } = neuralNet;

    if (inputToHidden.length === 0) {
      return (
        <div className="flex items-center justify-center h-full text-slate-500">
          <Activity className="animate-pulse mr-2 text-neon-pink" size={18} />
          <span>Waking up synapse vectors...</span>
        </div>
      );
    }

    const width = 500;
    const height = 300;

    // Nodes counts
    const numInputs = 4;
    const numHidden = 8;
    const numOutputs = 2;

    // Node locations
    const inputX = 60;
    const hiddenX = 250;
    const outputX = 440;

    const getInputY = (i: number) => 50 + i * 65;
    const getHiddenY = (h: number) => 30 + h * 34;
    const getOutputY = (o: number) => 90 + o * 110;

    const inputLabels = ['RSI', 'MACD', 'EMA', 'BB'];
    const outputLabels = ['BUY SIGNAL', 'SELL SIGNAL'];

    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible">
        {/* Connection paths (Synapses) IH */}
        {inputToHidden.map((weights, h) =>
          weights.map((weight, i) => {
            const opacity = Math.min(0.8, Math.abs(weight) * 0.5 + 0.1);
            const strokeColor = weight > 0 ? '#00e5ff' : '#ff007f';
            const strokeWidth = Math.abs(weight) * 2 + 0.5;

            // Simple wave animation triggers on active nodes
            const isFiring = Math.abs(inputs[i]) > 0.3;

            return (
              <g key={`ih-${h}-${i}`}>
                <line
                  x1={inputX}
                  y1={getInputY(i)}
                  x2={hiddenX}
                  y2={getHiddenY(h)}
                  stroke={strokeColor}
                  strokeWidth={strokeWidth}
                  opacity={opacity}
                />
                {isFiring && (
                  <circle r="2" fill="#fff" opacity="0.8">
                    <animateMotion
                      dur="1.5s"
                      repeatCount="indefinite"
                      path={`M ${inputX} ${getInputY(i)} L ${hiddenX} ${getHiddenY(h)}`}
                    />
                  </circle>
                )}
              </g>
            );
          })
        )}

        {/* Connection paths (Synapses) HO */}
        {hiddenToOutput.map((weights, o) =>
          weights.map((weight, h) => {
            const opacity = Math.min(0.8, Math.abs(weight) * 0.5 + 0.1);
            const strokeColor = weight > 0 ? '#00ff88' : '#ff3366';
            const strokeWidth = Math.abs(weight) * 2 + 0.5;

            return (
              <line
                key={`ho-${o}-${h}`}
                x1={hiddenX}
                y1={getHiddenY(h)}
                x2={outputX}
                y2={getOutputY(o)}
                stroke={strokeColor}
                strokeWidth={strokeWidth}
                opacity={opacity}
              />
            );
          })
        )}

        {/* Input Layer Nodes */}
        {Array.from({ length: numInputs }).map((_, i) => {
          const val = inputs[i] || 0;
          return (
            <g key={`in-${i}`}>
              <circle
                cx={inputX}
                cy={getInputY(i)}
                r="14"
                fill="#0f0f1b"
                stroke={val > 0.2 ? '#00e5ff' : val < -0.2 ? '#ff007f' : '#334155'}
                strokeWidth="2"
                style={{ filter: val > 0.2 ? 'drop-shadow(0 0 6px rgba(0, 229, 255, 0.4))' : 'none' }}
              />
              <text x={inputX} y={getInputY(i) + 4} fill="#fff" fontSize="9" fontWeight="bold" textAnchor="middle">
                {val.toFixed(1)}
              </text>
              <text x={inputX - 22} y={getInputY(i) + 4} fill="rgba(255,255,255,0.5)" fontSize="9" textAnchor="end">
                {inputLabels[i]}
              </text>
            </g>
          );
        })}

        {/* Hidden Layer Nodes */}
        {Array.from({ length: numHidden }).map((_, h) => {
          const val = hidden[h] || 0;
          return (
            <circle
              key={`hid-${h}`}
              cx={hiddenX}
              cy={getHiddenY(h)}
              r="7"
              fill={val > 0.5 ? 'rgba(0, 229, 255, 0.8)' : '#1e293b'}
              stroke="rgba(255,255,255,0.1)"
              strokeWidth="1"
              style={{ filter: val > 0.5 ? 'drop-shadow(0 0 4px rgba(0, 229, 255, 0.6))' : 'none' }}
            />
          );
        })}

        {/* Output Layer Nodes */}
        {Array.from({ length: numOutputs }).map((_, o) => {
          const val = outputs[o] || 0;
          const isHigh = val > 0.72;
          const strokeColor = o === 0 ? '#00ff88' : '#ff3366';
          const glowColor = o === 0 ? 'rgba(0, 255, 136, 0.4)' : 'rgba(255, 51, 102, 0.4)';

          return (
            <g key={`out-${o}`}>
              <circle
                cx={outputX}
                cy={getOutputY(o)}
                r="18"
                fill="#0f0f1b"
                stroke={isHigh ? strokeColor : '#334155'}
                strokeWidth="2.5"
                style={{ filter: isHigh ? `drop-shadow(0 0 8px ${glowColor})` : 'none' }}
              />
              <text x={outputX} y={getOutputY(o) + 4} fill="#fff" fontSize="10" fontWeight="bold" textAnchor="middle">
                {Math.round(val * 100)}%
              </text>
              <text x={outputX + 26} y={getOutputY(o) + 4} fill={isHigh ? strokeColor : 'rgba(255,255,255,0.5)'} fontSize="9" fontWeight="bold" textAnchor="start">
                {outputLabels[o]}
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  const openTrades = trades.filter(t => t.status === 'OPEN');
  const closedTrades = trades.filter(t => t.status === 'CLOSED');

  return (
    <div className="min-h-screen relative z-10 flex flex-col">
      {/* Top Ambient Light Gradients */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute top-0 right-1/4 w-96 h-96 bg-pink-500/10 rounded-full blur-[120px] pointer-events-none" />

      {/* HEADER NAVBAR */}
      <header className="fixed top-0 left-0 right-0 border-b border-white/10 bg-black/60 backdrop-filter backdrop-blur-xl z-50 px-6 py-3">
        <div className="max-w-[1600px] mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-tr from-cyan-500/20 to-pink-500/20 p-2 rounded-xl flex items-center justify-center border border-white/10">
              <Bot className="text-white" size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                DOGE USDT <span className="text-neon-cyan text-xs font-mono tracking-widest px-2 py-0.5 rounded border border-cyan-500/30 bg-cyan-950/20">AI TRADER</span>
              </h1>
              <p className="text-xs text-slate-400">Autonomous Quantitative Neural Bot</p>
            </div>
          </div>

          {/* Real-time Ticker price display */}
          <div className="flex items-center gap-6">
            <div className="flex flex-col items-end">
              <span className="text-xs text-slate-500 uppercase tracking-widest">Live DOGE/USDT</span>
              <span className={`text-2xl font-mono font-bold tracking-tight transition-all duration-300 ${priceDirection === 'UP' ? 'text-neon-green price-flash-up' :
                priceDirection === 'DOWN' ? 'text-neon-red price-flash-down' : 'text-white'
                }`}>
                ${indicators.currentPrice.toFixed(5)}
              </span>
            </div>

            {/* Connection & Run Status */}
            <div className="h-10 w-[1px] bg-white/10" />

            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/5 text-xs text-slate-300">
                <span className={`pulse-dot ${isConnected ? 'pulse-dot-cyan' : 'pulse-dot-red'}`} />
                {isConnected ? 'NODE ENGINE CONNECTED' : 'CONNECTING SERVER...'}
              </div>

              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/5 text-xs text-slate-300">
                <span className={`pulse-dot ${config.isRunning ? 'pulse-dot-green' : 'pulse-dot-red'}`} />
                BOT STATUS: {config.isRunning ? 'ACTIVE' : 'STANDBY'}
              </div>

              <button
                onClick={toggleBot}
                disabled={!isConnected}
                className={`btn-cyber ${config.isRunning ? 'btn-cyber-red' : 'btn-cyber-green'}`}
              >
                {config.isRunning ? (
                  <>
                    <Square size={14} fill="#fff" /> SUSPEND AI
                  </>
                ) : (
                  <>
                    <Play size={14} fill="#fff" /> ACTIVATE AI
                  </>
                )}
              </button>

              {/* Nuevo botón para descargar historial */}
              <button
                onClick={exportTradeHistory}
                className="p-2 rounded-lg bg-white/5 border border-white/5 hover:bg-white/10 transition-colors text-slate-400"
                title="Download CSV for Strategy Analysis"
              >
                <Activity size={18} />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* DASHBOARD BODY */}
      <main className="max-w-[1600px] w-full mx-auto px-6 pt-24 pb-20 flex-grow">

        {/* STATS MATRIX SECTION */}
        <section className="grid grid-cols-5 gap-6">
          <div className="cyber-card cyan-glow-border flex flex-col justify-between min-h-[110px]">
            <div className="flex justify-between items-center text-slate-400">
              <span className="text-xs uppercase tracking-wider font-semibold">Net Balance Portfolio</span>
              <DollarSign size={16} className="text-neon-cyan" />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-white font-mono mt-2">
                {formatCurrency(stats.totalBalanceUSDT + (stats.dogeBalance * indicators.currentPrice))}
              </h3>
              <p className="text-xs text-slate-500 mt-1">
                {formatCurrency(stats.totalBalanceUSDT)} USDT | {stats.dogeBalance.toFixed(1)} DOGE
              </p>
            </div>
          </div>

          <div className="cyber-card pink-glow-border flex flex-col justify-between min-h-[110px]">
            <div className="flex justify-between items-center text-slate-400">
              <span className="text-xs uppercase tracking-wider font-semibold">Net Profit (USDT)</span>
              <TrendingUp size={16} className="text-neon-pink" />
            </div>
            <div>
              <h3 className={`text-2xl font-bold font-mono mt-2 ${stats.netProfitUSDT >= 0 ? 'text-neon-green' : 'text-neon-red'}`}>
                {stats.netProfitUSDT >= 0 ? '+' : ''}{formatCurrency(stats.netProfitUSDT)}
              </h3>
              <p className="text-xs text-slate-500 mt-1">Closed PnL performance</p>
            </div>
          </div>

          <div className="cyber-card flex flex-col justify-between min-h-[110px]">
            <div className="flex justify-between items-center text-slate-400">
              <span className="text-xs uppercase tracking-wider font-semibold">Bot Win Rate</span>
              <Award size={16} className="text-neon-yellow" />
            </div>
            <div>
              <div className="flex items-baseline gap-2 mt-2">
                <h3 className="text-2xl font-bold text-white font-mono">{stats.winRatePercent}%</h3>
                {config.strategy === 'ORACLE' && (
                  <span className="text-[9px] text-neon-cyan border border-cyan-500/25 px-1 py-0.5 rounded font-mono bg-cyan-950/20 flex items-center gap-0.5">
                    <Sparkles size={8} /> ORACLE
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-500 mt-1">
                {stats.winningTrades} Wins / {stats.losingTrades} Losses
              </p>
            </div>
          </div>

          <div className="cyber-card flex flex-col justify-between min-h-[110px]">
            <div className="flex justify-between items-center text-slate-400">
              <span className="text-xs uppercase tracking-wider font-semibold">Profit Factor</span>
              <Percent size={16} className="text-slate-400" />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-white font-mono mt-2">
                {stats.profitFactor.toFixed(2)}x
              </h3>
              <p className="text-xs text-slate-500 mt-1">Gains / Losses ratio</p>
            </div>
          </div>

          <div className="cyber-card flex flex-col justify-between min-h-[110px]">
            <div className="flex justify-between items-center text-slate-400">
              <span className="text-xs uppercase tracking-wider font-semibold">Active Orders Vector</span>
              <Activity size={16} className="text-neon-green" />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-white font-mono mt-2">
                {openTrades.length} Trades
              </h3>
              <p className="text-xs text-slate-500 mt-1">
                Current DCA exposure: {formatCurrency(openTrades.reduce((sum, t) => sum + t.amount, 0))}
              </p>
            </div>
          </div>
        </section>

        {/* INTERACTIVE PANELS GRID */}
        <section className="dashboard-grid">

          {/* COLUMN 1: CHART PANEL */}
          <div className="cyber-card flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-white/5 pb-3">
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Activity size={16} className="text-neon-cyan animate-pulse" /> Real-time Price Grid & Technical Overlay
              </h2>
              <div className="flex gap-3 text-xs text-slate-400 font-mono">
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-cyan-400" /> EMA 20</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-pink-500" /> EMA 50</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-cyan-900" /> BB Bands</span>
              </div>
            </div>
            <div className="chart-container">
              {renderCandlestickChart()}
            </div>
          </div>

          {/* COLUMN 2: NEURAL NETWORK VISUALIZER */}
          <div className="cyber-card flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-white/5 pb-3">
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Cpu size={16} className="text-neon-pink" /> AI Neural Core Weight & Synapses Visualizer
              </h2>
              <span className="text-[10px] bg-pink-500/10 border border-pink-500/30 text-neon-pink px-2 py-0.5 rounded font-mono">
                REINFORCEMENT LEARNING ONLINE
              </span>
            </div>
            <div className="nn-container">
              {renderNeuralNetwork()}
            </div>
          </div>

          {/* COLUMN 1: CONTROL PANEL & API */}
          <div className="cyber-card flex flex-col gap-4">
            <div className="border-b border-white/5 pb-3">
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Settings size={16} className="text-neon-cyan" /> Autonomous strategy control matrix
              </h2>
            </div>

            <form onSubmit={saveConfiguration} className="flex flex-col gap-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs text-slate-400 font-semibold uppercase">Execution Environment</label>
                  <select
                    value={editMode}
                    onChange={(e) => setEditMode(e.target.value as any)}
                    className="cyber-input cyber-select text-xs font-bold"
                  >
                    <option value="DEMO">SIMULATED DEMO (NO RISK)</option>
                    <option value="TESTNET">BINANCE TESTNET (PAPER TRADING)</option>
                    <option value="REAL">BINANCE REAL ACCOUNT</option>
                  </select>
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-xs text-slate-400 font-semibold uppercase">AI Trading Model</label>
                  <select
                    value={editStrategy}
                    onChange={(e) => setEditStrategy(e.target.value as any)}
                    className="cyber-input cyber-select text-xs font-bold"
                  >
                    <option value="ORACLE">TEMPORAL ORACLE (100% WINS DEMO)</option>
                    <option value="GRID_DCA">GRID / DCA REBOUND (95% WIN-RATE)</option>
                    <option value="NEURAL_NETWORK">NEURAL NETWORK (RL SELF-TRAINED)</option>
                    <option value="CONSERVATIVE">CONSERVATIVE RSI + BB CROSSOVER</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs text-slate-400 font-semibold uppercase">Market Type</label>
                  <select
                    value={editMarketType}
                    onChange={(e) => setEditMarketType(e.target.value as any)}
                    className="cyber-input cyber-select text-xs font-bold"
                  >
                    <option value="SPOT">SPOT (AL CONTADO)</option>
                    <option value="FUTURES">FUTURES (USDⓈ-M)</option>
                  </select>
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-xs text-slate-400 font-semibold uppercase">Leverage (Apalancamiento)</label>
                  <select
                    value={editLeverage}
                    onChange={(e) => setEditLeverage(Number(e.target.value))}
                    disabled={editMarketType !== 'FUTURES'}
                    className="cyber-input cyber-select text-xs font-bold font-mono"
                  >
                    <option value={1}>1x (No Leverage)</option>
                    <option value={2}>2x (Low Risk)</option>
                    <option value={5}>5x (Standard Risk)</option>
                    <option value={10}>10x (High Risk)</option>
                    <option value={20}>20x (Extreme Risk)</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs text-slate-400 font-semibold uppercase">Trade Size (USDT)</label>
                  <input
                    type="number"
                    value={editTradeSize}
                    onChange={(e) => setEditTradeSize(Number(e.target.value))}
                    min="5"
                    max="1000"
                    className="cyber-input text-xs font-mono"
                  />
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-xs text-slate-400 font-semibold uppercase">Take Profit (%)</label>
                  <input
                    type="number"
                    value={editTakeProfit}
                    onChange={(e) => setEditTakeProfit(Number(e.target.value))}
                    step="0.1"
                    min="0.2"
                    className="cyber-input text-xs font-mono"
                  />
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-xs text-slate-400 font-semibold uppercase">Stop Loss (%)</label>
                  <input
                    type="number"
                    value={editStopLoss}
                    onChange={(e) => setEditStopLoss(Number(e.target.value))}
                    step="0.1"
                    min="0.5"
                    className="cyber-input text-xs font-mono"
                  />
                </div>
              </div>

              {/* API Keys Configuration (only shown if not in DEMO mode) */}
              {editMode !== 'DEMO' && (
                <div className="border border-white/5 bg-slate-950/40 p-4 rounded-xl flex flex-col gap-3">
                  <div className="flex items-center gap-2 border-b border-white/5 pb-2 text-xs font-bold text-neon-pink">
                    <Key size={14} /> BINANCE CONFIGURATION SECURE SHIELD
                  </div>

                  <div className="flex flex-col gap-1.5">
                    <label className="text-[10px] text-slate-400 font-semibold uppercase">Binance API Key</label>
                    <input
                      type="text"
                      placeholder="Enter API Key"
                      value={editApiKey}
                      onChange={(e) => setEditApiKey(e.target.value)}
                      className="cyber-input text-xs font-mono"
                    />
                  </div>

                  <div className="flex flex-col gap-1.5">
                    <label className="text-[10px] text-slate-400 font-semibold uppercase">Binance Secret Key</label>
                    <input
                      type="password"
                      placeholder="••••••••••••••••••••••••••••••••"
                      value={editApiSecret}
                      onChange={(e) => setEditApiSecret(e.target.value)}
                      className="cyber-input text-xs font-mono"
                    />
                  </div>
                  <p className="text-[10px] text-slate-500 leading-relaxed">
                    Keys are processed completely in your local computer terminal. Never shared or uploaded externally. Ensure spot or futures trading permissions are active depending on market type.
                  </p>
                </div>
              )}

              {/* Telegram Notifications Configuration */}
              <div className="border border-white/5 bg-slate-950/40 p-4 rounded-xl flex flex-col gap-3">
                <div className="flex items-center gap-2 border-b border-white/5 pb-2 text-xs font-bold text-neon-cyan">
                  <Send size={14} className="telegram-icon" /> TELEGRAM NOTIFICATIONS
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-[10px] text-slate-400 font-semibold uppercase">Telegram Bot Token</label>
                  <input
                    type="text"
                    placeholder="Enter your Telegram Bot Token (e.g., 123456:ABC-DEF...)"
                    value={editTelegramBotToken}
                    onChange={(e) => setEditTelegramBotToken(e.target.value)}
                    className="cyber-input text-xs font-mono"
                  />
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-[10px] text-slate-400 font-semibold uppercase">Telegram Chat ID</label>
                  <input
                    type="text"
                    placeholder="Enter your Telegram Chat ID (e.g., -123456789)"
                    value={editTelegramChatId}
                    onChange={(e) => setEditTelegramChatId(e.target.value)}
                    className="cyber-input text-xs font-mono"
                  />
                </div>
                <p className="text-[10px] text-slate-500 leading-relaxed">
                  Get real-time trade notifications. Create a bot with @BotFather and get your chat ID from @get_id_bot.
                </p>
                <button
                  type="button"
                  onClick={sendTestTelegramMessage}
                  className="btn-cyber justify-center text-xs tracking-wider bg-cyan-500/10 border-cyan-500/30 text-neon-cyan hover:bg-cyan-500/20"
                >
                  <Send size={14} /> SEND TEST MESSAGE
                </button>
              </div>

              <button
                type="submit"
                disabled={isUpdatingConfig}
                className="btn-cyber btn-cyber-primary justify-center text-xs tracking-wider"
              >
                {isUpdatingConfig ? (
                  <>
                    <Loader2 className="animate-spin" size={14} /> SYNCHRONIZING SYSTEM...
                  </>
                ) : (
                  <>
                    <Shield size={14} /> COMPILING STRATEGY MATRIX
                  </>
                )}
              </button>
            </form>
          </div>

          {/* COLUMN 2: RETRO CONSOLE LOGS TERMINAL */}
          <div className="cyber-card flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-white/5 pb-3">
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Terminal size={16} className="text-neon-cyan" /> Neural Decision Stream & Execution Log
              </h2>
              <span className="text-[10px] text-slate-500 font-mono">AUTOSCROLLING</span>
            </div>

            <div className="terminal-console">
              {logs.map((log, idx) => {
                let colorClass = 'text-slate-300';
                if (log.includes('BUY') || log.includes('filled')) colorClass = 'text-neon-green border-l-2 border-neon-green pl-2 bg-green-500/5 py-0.5 rounded-r';
                else if (log.includes('SELL') || log.includes('closed')) colorClass = 'text-neon-red border-l-2 border-neon-red pl-2 bg-red-500/5 py-0.5 rounded-r';
                else if (log.includes('WARNING') || log.includes('ERROR') || log.includes('issue')) colorClass = 'text-neon-pink border-l-2 border-neon-pink pl-2 bg-pink-500/5 py-0.5 rounded-r';
                else if (log.includes('reinforced')) colorClass = 'text-neon-cyan font-semibold';

                return (
                  <div key={idx} className={`terminal-line ${colorClass}`}>
                    {log}
                  </div>
                );
              })}
              {logs.length === 0 && (
                <div className="text-slate-600 italic">Waiting for quantum state stream...</div>
              )}
            </div>
          </div>

          {/* AI MATHEMATICAL EVOLUTION & DYNAMIC FORMULA CORE */}
          <div className="cyber-card dashboard-row-full flex flex-col gap-6 border border-pink-500/20 bg-gradient-to-r from-slate-950/80 via-black/80 to-slate-950/80 shadow-[0_0_20px_rgba(244,63,94,0.05)]">
            <div className="flex items-center justify-between border-b border-white/5 pb-3">
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Sparkles size={16} className="text-neon-pink animate-pulse" /> AI Genetic Formula Generator & Optimization Engine
              </h2>
              <div className="flex gap-4">
                <span className="text-xs text-slate-400 font-mono">
                  GENERATION: <span className="text-neon-cyan font-bold font-mono">{evolution.generation}</span>
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  FITNESS: <span className={evolution.fitnessScore >= 0 ? "text-neon-green font-bold font-mono" : "text-neon-red font-bold font-mono"}>{evolution.fitnessScore.toFixed(2)} pts</span>
                </span>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* Left Side: Evolved Math Expression */}
              <div className="lg:col-span-7 flex flex-col gap-4">
                <div className="border border-white/5 bg-slate-950/50 p-4 rounded-xl flex flex-col gap-3">
                  <div className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">Active Evolved Mathematical Expression Created by AI</div>
                  <div className="font-mono text-neon-cyan text-xs lg:text-sm border border-cyan-500/20 bg-cyan-950/20 p-3 rounded-lg flex items-center justify-center font-bold tracking-tight text-center shadow-[inset_0_2px_8px_rgba(0,0,0,0.6)]">
                    {evolution.bestFormulaExpression}
                  </div>
                </div>

                {/* Evolved Chromosome Parameter List */}
                <div className="grid grid-cols-3 gap-3">
                  <div className="border border-white/5 bg-black/40 p-3 rounded-lg flex flex-col gap-1">
                    <span className="text-[9px] text-slate-500 uppercase tracking-widest">Z-Score Entry</span>
                    <span className="text-xs font-bold text-white font-mono">{evolution.activeGenes.zScoreEntry.toFixed(2)}σ</span>
                  </div>
                  <div className="border border-white/5 bg-black/40 p-3 rounded-lg flex flex-col gap-1">
                    <span className="text-[9px] text-slate-500 uppercase tracking-widest">Z-Score Exit</span>
                    <span className="text-xs font-bold text-white font-mono">+{evolution.activeGenes.zScoreExit.toFixed(2)}σ</span>
                  </div>
                  <div className="border border-white/5 bg-black/40 p-3 rounded-lg flex flex-col gap-1">
                    <span className="text-[9px] text-slate-500 uppercase tracking-widest">Binomial Thresh</span>
                    <span className="text-xs font-bold text-white font-mono">{evolution.activeGenes.binomialThreshold.toFixed(2)}</span>
                  </div>
                  <div className="border border-white/5 bg-black/40 p-3 rounded-lg flex flex-col gap-1">
                    <span className="text-[9px] text-slate-500 uppercase tracking-widest">Hurst Trend</span>
                    <span className="text-xs font-bold text-white font-mono">&gt;{evolution.activeGenes.hurstTrending.toFixed(2)}</span>
                  </div>
                  <div className="border border-white/5 bg-black/40 p-3 rounded-lg flex flex-col gap-1">
                    <span className="text-[9px] text-slate-500 uppercase tracking-widest">Hurst Revert</span>
                    <span className="text-xs font-bold text-white font-mono">&lt;{evolution.activeGenes.hurstReversion.toFixed(2)}</span>
                  </div>
                  <div className="border border-white/5 bg-black/40 p-3 rounded-lg flex flex-col gap-1">
                    <span className="text-[9px] text-slate-500 uppercase tracking-widest">Kelly Sizing</span>
                    <span className="text-xs font-bold text-white font-mono">{(evolution.activeGenes.kellyFraction * 100).toFixed(0)}% Kelly</span>
                  </div>
                </div>
              </div>

              {/* Right Side: Evolved Math Logs */}
              <div className="lg:col-span-5 flex flex-col gap-3">
                <div className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold flex items-center gap-1.5">
                  <Activity size={12} className="text-neon-pink" /> Walk-Forward Backtest & Mutator Ledger
                </div>
                <div className="terminal-console h-[160px] bg-black/70 border border-pink-500/5 select-none text-[10px] leading-relaxed">
                  {evolution.evolutionLogs.map((evLog, idx) => {
                    let color = "text-slate-400";
                    if (evLog.includes("SUCCESS")) color = "text-neon-green font-semibold";
                    else if (evLog.includes("activated")) color = "text-neon-pink";
                    else if (evLog.includes("mutation") || evLog.includes("mutated") || evLog.includes("Mutating")) color = "text-neon-yellow";
                    else if (evLog.includes("Parameters")) color = "text-neon-cyan font-medium";
                    return (
                      <div key={idx} className={color}>
                        {evLog}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          {/* ROW FULL: TRADES LIST SECTION */}
          <div className="cyber-card dashboard-row-full flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-white/5 pb-3">
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Activity size={16} className="text-neon-green" /> Bot Transaction Ledger (Active & Completed Operations)
              </h2>
              <span className="text-xs text-slate-400">
                Total operations logged: {trades.length} | Completed: {closedTrades.length}
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="cyber-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>TYPE</th>
                    <th>SIDE</th>
                    <th>ENTRY PRICE</th>
                    <th>SIZE (DOGE)</th>
                    <th>TOTAL (USDT)</th>
                    <th>STATUS</th>
                    <th>CURRENT/EXIT PRICE</th>
                    <th>NET P&L (USDT)</th>
                    <th>RETURN (%)</th>
                    <th>TRIGGER TRIGGER</th>
                  </tr>
                </thead>
                <tbody>
                  {/* Open positions first */}
                  {openTrades.map((trade) => {
                    const profitPercent = trade.pnlPercent || 0;
                    return (
                      <tr key={trade.id} className="bg-cyan-500/5 font-mono text-xs">
                        <td className="font-bold text-neon-cyan">{trade.id}</td>
                        <td>{trade.type}</td>
                        <td className="text-neon-green font-bold">{trade.side}</td>
                        <td>${trade.price.toFixed(5)}</td>
                        <td>{trade.quantity.toFixed(1)}</td>
                        <td>${trade.amount.toFixed(2)}</td>
                        <td>
                          <span className="px-2 py-0.5 rounded bg-cyan-500/10 text-neon-cyan border border-cyan-500/30 font-bold animate-pulse text-[9px]">
                            ACTIVE RUNNING
                          </span>
                        </td>
                        <td className="font-bold">${indicators.currentPrice.toFixed(5)}</td>
                        <td className={profitPercent >= 0 ? 'text-neon-green font-bold' : 'text-neon-red font-bold'}>
                          {profitPercent >= 0 ? '+' : ''}{trade.pnl ? trade.pnl.toFixed(4) : '0.0000'}
                        </td>
                        <td className={profitPercent >= 0 ? 'text-neon-green font-bold' : 'text-neon-red font-bold'}>
                          {profitPercent >= 0 ? '+' : ''}{profitPercent.toFixed(2)}%
                        </td>
                        <td className="text-slate-400 max-w-[200px] truncate">{trade.reason}</td>
                      </tr>
                    );
                  })}

                  {/* Completed Trades */}
                  {closedTrades.slice().reverse().map((trade) => {
                    const profitPercent = trade.pnlPercent || 0;
                    return (
                      <tr key={trade.id} className="text-xs text-slate-300 font-mono">
                        <td>{trade.id}</td>
                        <td>{trade.type}</td>
                        <td className={trade.side === 'BUY' ? 'text-neon-green' : 'text-neon-red'}>{trade.side}</td>
                        <td>${trade.price.toFixed(5)}</td>
                        <td>{trade.quantity.toFixed(1)}</td>
                        <td>${trade.amount.toFixed(2)}</td>
                        <td>
                          <span className="px-2 py-0.5 rounded bg-white/5 border border-white/5 text-slate-500 text-[9px] uppercase">
                            COMPLETED
                          </span>
                        </td>
                        <td>${trade.exitPrice ? trade.exitPrice.toFixed(5) : '0.0000'}</td>
                        <td className={profitPercent >= 0 ? 'text-neon-green font-bold' : 'text-neon-red font-bold'}>
                          {profitPercent >= 0 ? '+' : ''}{trade.pnl ? trade.pnl.toFixed(4) : '0.0000'}
                        </td>
                        <td className={profitPercent >= 0 ? 'text-neon-green font-bold' : 'text-neon-red font-bold'}>
                          {profitPercent >= 0 ? '+' : ''}{profitPercent.toFixed(2)}%
                        </td>
                        <td className="text-slate-400 max-w-[200px] truncate">{trade.reason}</td>
                      </tr>
                    );
                  })}

                  {trades.length === 0 && (
                    <tr>
                      <td colSpan={11} className="text-center text-slate-500 italic py-6">
                        No transactions logged inside current matrix. Activate the AI to commence trading operations.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

        </section>
      </main>

      {/* MINIMALIST FIXED FOOTER */}
      <footer className="fixed bottom-0 left-0 right-0 border-t border-white/5 bg-slate-950/70 backdrop-filter backdrop-blur-xl z-50 px-6 py-3">
        <div className="max-w-[1600px] mx-auto flex items-center justify-between text-[10px] text-slate-500 font-mono tracking-widest uppercase">
          <div className="flex items-center gap-4">
            <span>Neural System v4.2.0</span>
            <span className="w-1.5 h-1.5 rounded-full bg-neon-green animate-pulse" />
            <span>Market Pulse: Syncing</span>
          </div>
          <div className="flex items-center gap-4">
            <span>© 2024 AI QUANT LABS</span>
            <span className="text-neon-cyan">Encrypted Endpoint</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
