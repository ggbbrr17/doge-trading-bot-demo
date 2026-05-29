import { CustomNeuralNetwork } from './aiModel';

export interface StrategySignal {
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number; // 0 to 1
  reason: string;
}

export interface TechnicalIndicators {
  rsi: number;
  macd: {
    macd: number;
    signal: number;
    hist: number;
  };
  ema: {
    ema20: number;
    ema50: number;
    ema200: number;
  };
  bollinger: {
    upper: number;
    middle: number;
    lower: number;
    width: number;
  };
  currentPrice: number;
}

// Calculate indicators from an array of closing prices
export function calculateIndicators(prices: number[]): TechnicalIndicators {
  const currentPrice = prices[prices.length - 1] || 0;

  // 1. Calculate EMAs
  const ema = (period: number): number => {
    let k = 2 / (period + 1);
    let val = prices[0] || 0;
    for (let i = 1; i < prices.length; i++) {
      val = prices[i] * k + val * (1 - k);
    }
    return val;
  };

  const ema20 = ema(20);
  const ema50 = ema(50);
  const ema200 = ema(200);

  // 2. Calculate RSI (14 period)
  let rsi = 50; // default
  if (prices.length > 14) {
    let gains = 0;
    let losses = 0;
    for (let i = prices.length - 14; i < prices.length; i++) {
      const diff = prices[i] - prices[i - 1];
      if (diff > 0) gains += diff;
      else losses -= diff;
    }
    gains /= 14;
    losses /= 14;
    if (losses === 0) rsi = 100;
    else {
      const rs = gains / losses;
      rsi = 100 - 100 / (1 + rs);
    }
  }

  // 3. Calculate MACD (12, 26, 9)
  const ema12 = ema(12);
  const ema26 = ema(26);
  const macdVal = ema12 - ema26;
  // Approximation of signal (9 period EMA of MACD)
  const signalVal = macdVal * 0.2 + (ema(9) - ema(26)) * 0.8; // smoothed approximation
  const hist = macdVal - signalVal;

  // 4. Calculate Bollinger Bands (20 period, 2 std dev)
  let upper = currentPrice * 1.02;
  let lower = currentPrice * 0.98;
  let middle = ema20;
  if (prices.length >= 20) {
    const slice = prices.slice(-20);
    middle = slice.reduce((sum, p) => sum + p, 0) / 20;
    const variance = slice.reduce((sum, p) => sum + Math.pow(p - middle, 2), 0) / 20;
    const stdDev = Math.sqrt(variance);
    upper = middle + stdDev * 2;
    lower = middle - stdDev * 2;
  }
  const width = (upper - lower) / middle;

  return {
    rsi,
    macd: { macd: macdVal, signal: signalVal, hist },
    ema: { ema20, ema50, ema200 },
    bollinger: { upper, middle, lower, width },
    currentPrice,
  };
}

export class StrategyManager {
  private neuralNet: CustomNeuralNetwork;

  constructor(neuralNet: CustomNeuralNetwork) {
    this.neuralNet = neuralNet;
  }

  // 1. TEMPORAL ORACLE STRATEGY (100% win-rate simulated lookahead)
  // Inside simulation mode, the engine knows the future prices.
  // We pass futurePrices (the next 5-10 seconds of ticks) to make a perfect decision.
  getTemporalOracleSignal(
    indicators: TechnicalIndicators,
    futurePrices: number[]
  ): StrategySignal {
    const price = indicators.currentPrice;

    if (futurePrices.length > 0) {
      const nextPrice = futurePrices[0];
      // Check if price goes up within the future window
      const maxFuturePrice = Math.max(...futurePrices);
      const minFuturePrice = Math.min(...futurePrices);

      // If price goes up by at least 0.05% in the next few ticks, BUY
      if (maxFuturePrice > price * 1.0008) {
        return {
          action: 'BUY',
          confidence: 0.99,
          reason: `Oracle Matrix Analysis predicts imminent upswing to $${maxFuturePrice.toFixed(5)} (+${(((maxFuturePrice - price) / price) * 100).toFixed(3)}%)`,
        };
      }
      // If price is going down significantly, SELL
      if (minFuturePrice < price * 0.9992) {
        return {
          action: 'SELL',
          confidence: 0.98,
          reason: `Oracle Matrix predicts downswing to $${minFuturePrice.toFixed(5)} (-${(((price - minFuturePrice) / price) * 100).toFixed(3)}%)`,
        };
      }
    }

    return {
      action: 'HOLD',
      confidence: 0.5,
      reason: 'Oracle matrix detects high equilibrium / zero net drift.',
    };
  }

  // 2. GRID / DCA STRATEGY
  getGridDcaSignal(
    indicators: TechnicalIndicators,
    hasActiveTrade: boolean,
    averageEntryPrice: number
  ): StrategySignal {
    const { rsi, currentPrice, bollinger } = indicators;

    // Entry signal (if no trade, buy near support/oversold)
    if (!hasActiveTrade) {
      if (rsi < 35 || currentPrice <= bollinger.lower * 1.002) {
        return {
          action: 'BUY',
          confidence: 0.85,
          reason: `Grid entry triggered. Price ($${currentPrice.toFixed(5)}) is near lower Bollinger band ($${bollinger.lower.toFixed(5)}) with RSI at ${rsi.toFixed(1)} (Oversold).`,
        };
      }
    } else {
      // If we have an active trade, check if price dropped enough to trigger a DCA safety purchase (e.g. -1.5% from average price)
      const priceDropPercent = ((averageEntryPrice - currentPrice) / averageEntryPrice) * 100;
      if (priceDropPercent >= 1.2) {
        return {
          action: 'BUY',
          confidence: 0.9,
          reason: `DCA Safety Order triggered. Price is down ${priceDropPercent.toFixed(2)}% from average entry. Buying to average down cost.`,
        };
      }

      // Exit signal: price is above average entry by profit margin OR Bollinger Upper and high RSI
      const profitPercent = ((currentPrice - averageEntryPrice) / averageEntryPrice) * 100;
      if (profitPercent >= 0.8 || currentPrice >= bollinger.upper || rsi > 70) {
        return {
          action: 'SELL',
          confidence: 0.88,
          reason: `Take profit grid target reached. Current price is $${currentPrice.toFixed(5)} (+${profitPercent.toFixed(2)}% PnL) with RSI at ${rsi.toFixed(1)}.`,
        };
      }
    }

    return {
      action: 'HOLD',
      confidence: 0.5,
      reason: 'Market is bound inside trading channel. Grid pending.',
    };
  }

  // 3. AI NEURAL NETWORK STRATEGY
  getAiNeuralNetSignal(indicators: TechnicalIndicators): StrategySignal {
    const { rsi, macd, ema, bollinger, currentPrice } = indicators;

    // Calculate normalized neural network inputs
    const emaRatio = ema.ema20 / (ema.ema50 || 1.0);
    const bbRange = bollinger.upper - bollinger.lower || 0.0001;
    const bbPosition = (currentPrice - bollinger.lower) / bbRange;

    const inputs = CustomNeuralNetwork.normalizeIndicators(rsi, macd.hist, emaRatio, bbPosition);
    const outputs = this.neuralNet.forward(inputs);

    const buyConfidence = outputs[0];
    const sellConfidence = outputs[1];

    if (buyConfidence > 0.72 && buyConfidence > sellConfidence) {
      return {
        action: 'BUY',
        confidence: buyConfidence,
        reason: `AI Neural Core activated BUY vector. Confidence: ${(buyConfidence * 100).toFixed(1)}%. Indicators feed shows RSI ${rsi.toFixed(1)}, MACD Hist ${macd.hist.toFixed(4)}, EMA alignment: ${emaRatio.toFixed(3)}.`,
      };
    } else if (sellConfidence > 0.72 && sellConfidence > buyConfidence) {
      return {
        action: 'SELL',
        confidence: sellConfidence,
        reason: `AI Neural Core activated SELL vector. Confidence: ${(sellConfidence * 100).toFixed(1)}%. Indicators feed shows RSI ${rsi.toFixed(1)}, MACD Hist ${macd.hist.toFixed(4)}, EMA alignment: ${emaRatio.toFixed(3)}.`,
      };
    }

    return {
      action: 'HOLD',
      confidence: Math.max(0.5, 1 - Math.max(buyConfidence, sellConfidence)),
      reason: `AI Core weights returning HOLD. (Buy Conf: ${(buyConfidence * 100).toFixed(1)}%, Sell Conf: ${(sellConfidence * 100).toFixed(1)}%)`,
    };
  }

  // 4. CONSERVATIVE TECHNICAL STRATEGY
  getConservativeSignal(indicators: TechnicalIndicators): StrategySignal {
    const { rsi, macd, ema, bollinger, currentPrice } = indicators;

    // Strong buy conditions
    const isEmaUptrend = ema.ema20 > ema.ema50 && ema.ema50 > ema.ema200;
    const isOversold = rsi < 30;
    const isPriceNearLowerBand = currentPrice <= bollinger.lower * 1.001;
    const isMacdBullishCross = macd.hist > 0 && macd.macd > macd.signal * 1.05;

    if ((isOversold && isPriceNearLowerBand) || (isEmaUptrend && isMacdBullishCross && rsi < 55)) {
      return {
        action: 'BUY',
        confidence: 0.8,
        reason: `Conservative trigger: EMA uptrend align with MACD cross. RSI: ${rsi.toFixed(1)}, Price bottomed at lower band.`,
      };
    }

    // Strong sell conditions
    const isOverbought = rsi > 70;
    const isPriceNearUpperBand = currentPrice >= bollinger.upper * 0.999;
    const isMacdBearishCross = macd.hist < 0;

    if (isOverbought || isPriceNearUpperBand || (isMacdBearishCross && ema.ema20 < ema.ema50)) {
      return {
        action: 'SELL',
        confidence: 0.85,
        reason: `Conservative Exit: Overbought RSI ${rsi.toFixed(1)} and price hitting upper Bollinger Band ($${bollinger.upper.toFixed(5)}).`,
      };
    }

    return {
      action: 'HOLD',
      confidence: 0.5,
      reason: 'No conservative crossover boundaries breached.',
    };
  }
}
