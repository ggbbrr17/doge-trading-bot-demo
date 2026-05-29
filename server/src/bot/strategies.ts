import { CustomNeuralNetwork } from './aiModel';
import { MathGenes } from './evolutionEngine';

export interface StrategySignal {
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number; // 0 to 1
  reason: string;
}

export interface TechnicalIndicators {
  rsi: number;
  macd: { macd: number; signal: number; hist: number };
  ema: { ema20: number; ema50: number; ema200: number };
  bollinger: { upper: number; middle: number; lower: number; width: number };
  currentPrice: number;
}

// ═══════════════════════════════════════════════════════════════════════════════
// QUANTITATIVE MATHEMATICS LIBRARY
// Professional-grade formulas used by Renaissance Technologies, Citadel,
// Two Sigma, DE Shaw, and elite algorithmic trading desks worldwide.
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * LOG-FACTORIAL using Stirling's approximation.
 * Avoids floating-point overflow when computing large binomial coefficients.
 * ln(n!) ≈ n·ln(n) - n + 0.5·ln(2πn) + 1/(12n)
 */
function logFactorial(n: number): number {
  if (n <= 1) return 0;
  if (n <= 20) {
    let f = 0;
    for (let i = 2; i <= n; i++) f += Math.log(i);
    return f;
  }
  return n * Math.log(n) - n + 0.5 * Math.log(2 * Math.PI * n) + 1 / (12 * n);
}

/**
 * LOG BINOMIAL COEFFICIENT: ln(C(n, k))
 * ln C(n,k) = ln(n!) - ln(k!) - ln((n-k)!)
 */
function logBinomCoeff(n: number, k: number): number {
  if (k < 0 || k > n) return -Infinity;
  return logFactorial(n) - logFactorial(k) - logFactorial(n - k);
}

/**
 * BINOMIAL PMF: P(X = k | n, p)
 * Probability of EXACTLY k up-moves in n price ticks.
 * Core of the ORACLE Binomial strategy.
 */
function binomialPMF(n: number, k: number, p: number): number {
  if (p <= 0) return k === 0 ? 1 : 0;
  if (p >= 1) return k === n ? 1 : 0;
  return Math.exp(logBinomCoeff(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p));
}

/**
 * BINOMIAL TAIL PROBABILITY: P(X >= k | n, p)
 * If this is SMALL (< 0.15), the recent run of up-ticks is statistically
 * improbable → mean reversion / exhaustion expected → SELL.
 * If P(X >= k) is LARGE (> 0.8), uptrend has statistical support → BUY.
 */
function binomialTailProbability(n: number, k: number, p: number): number {
  let prob = 0;
  for (let i = k; i <= n; i++) {
    prob += binomialPMF(n, i, p);
  }
  return Math.min(1, prob);
}

/**
 * COUNT UP-MOVES in the last n prices.
 * A "move" is considered "up" if current tick > previous tick.
 */
function countUpMoves(prices: number[]): { upMoves: number; total: number } {
  let upMoves = 0;
  const total = prices.length - 1;
  for (let i = 1; i < prices.length; i++) {
    if (prices[i] > prices[i - 1]) upMoves++;
  }
  return { upMoves, total };
}

/**
 * HURST EXPONENT via R/S Analysis
 * H > 0.55 → persistent/trending market (follow momentum)
 * H ≈ 0.50 → random walk (efficient market)
 * H < 0.45 → anti-persistent/mean-reverting (fade moves)
 */
function hurstExponent(prices: number[]): number {
  if (prices.length < 20) return 0.5;
  const logReturns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    if (prices[i - 1] > 0) {
      logReturns.push(Math.log(prices[i] / prices[i - 1]));
    }
  }
  if (logReturns.length < 10) return 0.5;
  const mean = logReturns.reduce((a, b) => a + b, 0) / logReturns.length;
  const deviations = logReturns.map(r => r - mean);
  const cumSum: number[] = [];
  let cs = 0;
  for (const d of deviations) { cs += d; cumSum.push(cs); }
  const R = Math.max(...cumSum) - Math.min(...cumSum);
  const variance = deviations.reduce((s, d) => s + d * d, 0) / deviations.length;
  const S = Math.sqrt(variance);
  if (S === 0 || R === 0) return 0.5;
  const H = Math.log(R / S) / Math.log(logReturns.length);
  return Math.max(0.1, Math.min(0.9, H));
}

/**
 * KALMAN FILTER (1D, constant velocity model)
 */
function kalmanFilter(prices: number[], noiseRatio = 0.005): { filtered: number[]; velocity: number; acceleration: number } {
  const Q = 0.0001; // Process noise
  const R_noise = noiseRatio; // Measurement noise

  let x = prices[0];
  let P = 1.0;
  const filtered: number[] = [x];

  for (let i = 1; i < prices.length; i++) {
    P = P + Q;
    const K = P / (P + R_noise); // Kalman gain
    x = x + K * (prices[i] - x);
    P = (1 - K) * P;
    filtered.push(x);
  }

  const n = filtered.length;
  const velocity = n >= 2 ? filtered[n - 1] - filtered[n - 2] : 0;
  const acceleration = n >= 3 ? (filtered[n - 1] - filtered[n - 2]) - (filtered[n - 2] - filtered[n - 3]) : 0;

  return { filtered, velocity, acceleration };
}

/**
 * Z-SCORE: Statistical distance of current price from its rolling mean.
 */
function zScore(prices: number[], window: number = 30): number {
  if (prices.length < window) return 0;
  const slice = prices.slice(-window);
  const mean = slice.reduce((a, b) => a + b, 0) / window;
  const variance = slice.reduce((s, p) => s + Math.pow(p - mean, 2), 0) / window;
  const stdDev = Math.sqrt(variance);
  if (stdDev === 0) return 0;
  return (prices[prices.length - 1] - mean) / stdDev;
}

/**
 * KELLY CRITERION - Mathematically optimal bet sizing
 */
function kellyCriterion(winRate: number, avgWin: number, avgLoss: number, fraction = 0.25): number {
  if (avgLoss <= 0 || winRate <= 0 || winRate >= 1) return 0;
  const b = avgWin / Math.abs(avgLoss);
  const p = winRate;
  const q = 1 - p;
  const kelly = (b * p - q) / b;
  return Math.max(-1, Math.min(1, kelly * fraction));
}

/**
 * VARIANCE RATIO TEST (Lo & MacKinlay, 1988)
 */
function varianceRatio(prices: number[], shortPeriod: number = 2, longPeriod: number = 10): number {
  if (prices.length < longPeriod + 1) return 1.0;
  const returns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    if (prices[i - 1] > 0) returns.push(Math.log(prices[i] / prices[i - 1]));
  }
  if (returns.length < longPeriod) return 1.0;

  const shortR = returns.slice(-shortPeriod);
  const longR = returns.slice(-longPeriod);
  const varShort = shortR.reduce((s, r) => s + r * r, 0) / shortPeriod;
  const varLong = longR.reduce((s, r) => s + r * r, 0) / longPeriod;

  if (varShort === 0) return 1.0;
  return varLong / (varShort * (longPeriod / shortPeriod));
}

/**
 * MOMENTUM FACTOR (Jegadeesh & Titman, 1993)
 */
function momentumScore(prices: number[], lookback: number = 20, skip: number = 1): number {
  if (prices.length < lookback + skip + 1) return 0;
  const start = prices[prices.length - lookback - skip];
  const end = prices[prices.length - 1 - skip];
  if (start === 0) return 0;
  return (end - start) / start;
}

// ═══════════════════════════════════════════════════════════════════════════════
// UI compatibility helper
// ═══════════════════════════════════════════════════════════════════════════════
export function calculateIndicators(prices: number[]): TechnicalIndicators {
  const currentPrice = prices[prices.length - 1] || 0;

  const emaCalc = (period: number): number => {
    let k = 2 / (period + 1);
    let val = prices[0] || 0;
    for (let i = 1; i < prices.length; i++) val = prices[i] * k + val * (1 - k);
    return val;
  };

  const ema20 = emaCalc(20);
  const ema50 = emaCalc(50);
  const ema200 = emaCalc(200);

  let rsi = 50;
  if (prices.length > 14) {
    let gains = 0, losses = 0;
    for (let i = prices.length - 14; i < prices.length; i++) {
      const diff = prices[i] - prices[i - 1];
      if (diff > 0) gains += diff; else losses -= diff;
    }
    gains /= 14; losses /= 14;
    if (losses === 0) rsi = 100;
    else { const rs = gains / losses; rsi = 100 - 100 / (1 + rs); }
  }

  const ema12 = emaCalc(12);
  const ema26 = emaCalc(26);
  const macdVal = ema12 - ema26;
  const signalVal = macdVal * 0.2 + (emaCalc(9) - ema26) * 0.8;
  const hist = macdVal - signalVal;

  let upper = currentPrice * 1.02, lower = currentPrice * 0.98, middle = ema20;
  if (prices.length >= 20) {
    const slice = prices.slice(-20);
    middle = slice.reduce((s, p) => s + p, 0) / 20;
    const variance = slice.reduce((s, p) => s + Math.pow(p - middle, 2), 0) / 20;
    const stdDev = Math.sqrt(variance);
    upper = middle + stdDev * 2;
    lower = middle - stdDev * 2;
  }
  const width = (upper - lower) / (middle || 1);

  return {
    rsi, macd: { macd: macdVal, signal: signalVal, hist },
    ema: { ema20, ema50, ema200 },
    bollinger: { upper, middle, lower, width },
    currentPrice,
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// STRATEGY MANAGER
// ═══════════════════════════════════════════════════════════════════════════════
export class StrategyManager {
  private neuralNet: CustomNeuralNetwork;

  constructor(neuralNet: CustomNeuralNetwork) {
    this.neuralNet = neuralNet;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // STRATEGY 1: BINOMIAL ORACLE (Evolved parameters)
  // ───────────────────────────────────────────────────────────────────────────
  getTemporalOracleSignal(
    indicators: TechnicalIndicators,
    futurePrices: number[],
    priceHistory: number[] = [],
    genes?: MathGenes
  ): StrategySignal {
    const prices = priceHistory.length > 0 ? priceHistory : [indicators.currentPrice];
    const N = Math.min(20, prices.length - 1);
    if (N < 5) {
      return { action: 'HOLD', confidence: 0.5, reason: 'Binomial Oracle: insufficient price history.' };
    }

    const recentPrices = prices.slice(-N - 1);
    const { upMoves, total } = countUpMoves(recentPrices);
    const downMoves = total - upMoves;

    // Use evolved momentum lookback gene
    const lookback = genes ? genes.momentumLookback : 20;
    const momentum = momentumScore(prices, Math.min(lookback, prices.length - 1));
    const p = Math.max(0.2, Math.min(0.8, 0.5 + momentum * 5));

    // Use evolved kalmanNoiseRatio
    const noiseRatio = genes ? genes.kalmanNoiseRatio : 0.005;
    const kalman = kalmanFilter(recentPrices, noiseRatio);
    const velocity = kalman.velocity;
    const acceleration = kalman.acceleration;

    const pUpTail = binomialTailProbability(total, upMoves, 0.5);
    const pDownTail = binomialTailProbability(total, downMoves, 0.5);

    const upPct = ((upMoves / total) * 100).toFixed(0);
    const pUpFmt = (pUpTail * 100).toFixed(1);
    const pDownFmt = (pDownTail * 100).toFixed(1);

    const binomialThreshold = genes ? genes.binomialThreshold : 0.70;

    // BUY: streak statistically supported by evolved tail probability & Kalman filter
    if (pUpTail > binomialThreshold && velocity > 0 && acceleration >= 0 && upMoves > downMoves) {
      const confidence = Math.min(0.97, 0.7 + (pUpTail - binomialThreshold) * 0.9);
      return {
        action: 'BUY',
        confidence,
        reason: `Binomial Oracle: ${upMoves}/${total} up-ticks (${upPct}%). P(run|H₀)=${pUpFmt}% — uptrend verified. Evolved Threshold: ${binomialThreshold.toFixed(2)}. Kalman velocity: +${(velocity * 10000).toFixed(4)}.`,
      };
    }

    // SELL: exhaustion or downward momentum confirmed
    if ((pUpTail < (1 - binomialThreshold) && upMoves > downMoves) || (pDownTail > (binomialThreshold + 0.05) && velocity < 0)) {
      const confidence = Math.min(0.96, 0.72 + (1 - pUpTail) * 0.25);
      return {
        action: 'SELL',
        confidence,
        reason: `Binomial Oracle: streak exhaustion. P(${upMoves} up|H₀)=${pUpFmt}% — mean reversion expected. Kalman velocity: ${(velocity * 10000).toFixed(4)}.`,
      };
    }

    if (pDownTail > (binomialThreshold + 0.10) && velocity < 0 && downMoves > upMoves) {
      const confidence = Math.min(0.95, 0.70 + (pDownTail - binomialThreshold) * 0.85);
      return {
        action: 'SELL',
        confidence,
        reason: `Binomial Oracle: bearish dominance. P(down run|H₀)=${pDownFmt}%. Evolved Threshold: ${binomialThreshold.toFixed(2)}.`,
      };
    }

    return {
      action: 'HOLD',
      confidence: 0.5,
      reason: `Binomial Oracle: no significant edge. Up=${upMoves}/${total} (P=${pUpFmt}%).`,
    };
  }

  // ───────────────────────────────────────────────────────────────────────────
  // STRATEGY 2: Z-SCORE STATISTICAL ARBITRAGE (DCA/GRID mode - Evolved parameters)
  // ───────────────────────────────────────────────────────────────────────────
  getGridDcaSignal(
    indicators: TechnicalIndicators,
    hasActiveTrade: boolean,
    averageEntryPrice: number,
    priceHistory: number[] = [],
    genes?: MathGenes
  ): StrategySignal {
    const prices = priceHistory.length > 0 ? priceHistory : [indicators.currentPrice];
    const z30 = zScore(prices, 30);
    const z15 = zScore(prices, 15);
    const currentPrice = indicators.currentPrice;

    const longPeriod = genes ? genes.varianceRatioLongPeriod : 10;
    const vr = varianceRatio(prices, 2, longPeriod);
    const isMeanReverting = vr < 0.92;

    const zEntry = genes ? genes.zScoreEntry : -2.0;
    const zExit = genes ? genes.zScoreExit : 1.5;

    const vrFmt = vr.toFixed(3);
    const z30Fmt = z30.toFixed(2);
    const z15Fmt = z15.toFixed(2);

    if (!hasActiveTrade) {
      // Entry timings based on evolved Z-score thresholds
      if (z30 < zEntry && z15 < (zEntry + 0.5) && isMeanReverting) {
        const confidence = Math.min(0.93, 0.75 + Math.abs(z30 - zEntry) * 0.06);
        return {
          action: 'BUY',
          confidence,
          reason: `Z-Score Stat Arb: price is ${Math.abs(z30).toFixed(2)}σ below mean (z₃₀=${z30Fmt}, z₁₅=${z15Fmt}). Evolved entry z=${zEntry.toFixed(2)}. VR=${vrFmt} (mean-reverting).`,
        };
      }
      if (z30 < (zEntry - 0.5)) {
        return {
          action: 'BUY',
          confidence: 0.80,
          reason: `Z-Score entry: extreme statistical dislocation z=${z30Fmt}σ. Evolved entry target: ${zEntry.toFixed(2)}.`,
        };
      }
    } else {
      // DCA safety order at extreme dislocation
      const dropPct = ((averageEntryPrice - currentPrice) / averageEntryPrice) * 100;
      if (z30 < (zEntry - 1.0) && dropPct > 1.5) {
        return {
          action: 'BUY',
          confidence: 0.88,
          reason: `Z-Score DCA: extreme deviation z=${z30Fmt}σ and price drop of -${dropPct.toFixed(2)}%. Averaging down.`,
        };
      }

      // Exit timing using evolved zScoreExit parameter
      const profitPct = ((currentPrice - averageEntryPrice) / averageEntryPrice) * 100;
      if (z30 > zExit || (z30 > (zExit - 0.7) && profitPct > 0.5)) {
        const confidence = Math.min(0.92, 0.75 + (z30 - zExit) * 0.08);
        return {
          action: 'SELL',
          confidence,
          reason: `Z-Score exit: price reverted to z=${z30Fmt}σ. Evolved Exit z=${zExit.toFixed(2)}. PnL: +${profitPct.toFixed(2)}%.`,
        };
      }

      if (z30 > (zExit + 1.0)) {
        return {
          action: 'SELL',
          confidence: 0.90,
          reason: `Z-Score overbought: extreme target reached at z=${z30Fmt}σ. Reversion expected.`,
        };
      }
    }

    return {
      action: 'HOLD',
      confidence: 0.5,
      reason: `Z-Score equilibrium: z₃₀=${z30Fmt}σ, z₁₅=${z15Fmt}σ. VR=${vrFmt}.`,
    };
  }

  // ───────────────────────────────────────────────────────────────────────────
  // STRATEGY 3: KALMAN FILTER + HURST EXPONENT (Evolved parameters)
  // ───────────────────────────────────────────────────────────────────────────
  getAiNeuralNetSignal(
    indicators: TechnicalIndicators,
    priceHistory: number[] = [],
    genes?: MathGenes
  ): StrategySignal {
    const prices = priceHistory.length > 0 ? priceHistory : [indicators.currentPrice];
    if (prices.length < 15) {
      return { action: 'HOLD', confidence: 0.5, reason: 'Kalman/Hurst: insufficient data.' };
    }

    const noiseRatio = genes ? genes.kalmanNoiseRatio : 0.005;
    const kalman = kalmanFilter(prices.slice(-50), noiseRatio);
    const { velocity, acceleration } = kalman;

    // Use evolved Hurst thresholds
    const H = hurstExponent(prices.slice(-40));
    const trendH = genes ? genes.hurstTrending : 0.55;
    const revertH = genes ? genes.hurstReversion : 0.45;

    const isTrending = H > trendH;
    const isMeanReverting = H < revertH;

    const lookback = genes ? genes.momentumLookback : 20;
    const mom20 = momentumScore(prices, Math.min(lookback, prices.length - 2));
    const mom10 = momentumScore(prices, Math.min(Math.round(lookback / 2), prices.length - 2));

    const z = zScore(prices, Math.min(25, prices.length));

    const HFmt = H.toFixed(3);
    const velFmt = (velocity * 10000).toFixed(4);
    const accFmt = (acceleration * 10000).toFixed(4);
    const momFmt = (mom20 * 100).toFixed(3);
    const regime = isTrending ? 'TRENDING' : isMeanReverting ? 'MEAN-REVERTING' : 'RANDOM WALK';

    if (isTrending) {
      if (velocity > 0 && acceleration >= 0 && mom20 > 0.001 && mom10 > 0) {
        const confidence = Math.min(0.94, 0.72 + H * 0.25 + mom20 * 5);
        return {
          action: 'BUY',
          confidence,
          reason: `Kalman+Hurst [${regime}]: H=${HFmt} confirms trend. Kalman velocity=+${velFmt}. Trend parameter=${trendH.toFixed(2)}.`,
        };
      }
      if (velocity < 0 && acceleration <= 0 && mom20 < -0.001) {
        const confidence = Math.min(0.93, 0.72 + H * 0.25 + Math.abs(mom20) * 5);
        return {
          action: 'SELL',
          confidence,
          reason: `Kalman+Hurst [${regime}]: H=${HFmt} confirms downtrend. Kalman velocity=${velFmt}. Exiting trending regime.`,
        };
      }
    } else if (isMeanReverting) {
      if (z < -1.8 && velocity > 0) {
        const confidence = Math.min(0.91, 0.73 + (1 - H) * 0.2);
        return {
          action: 'BUY',
          confidence,
          reason: `Kalman+Hurst [${regime}]: H=${HFmt} (reverting). Z-score=${z.toFixed(2)}σ. Kalman recovery velocity=+${velFmt}.`,
        };
      }
      if (z > 1.8 && velocity < 0) {
        const confidence = Math.min(0.90, 0.73 + (1 - H) * 0.2);
        return {
          action: 'SELL',
          confidence,
          reason: `Kalman+Hurst [${regime}]: H=${HFmt} (reverting). Z-score=+${z.toFixed(2)}σ. Kalman decline velocity=${velFmt}.`,
        };
      }
    }

    return {
      action: 'HOLD',
      confidence: 0.5,
      reason: `Kalman+Hurst [${regime}]: H=${HFmt}. Kalman vel=${velFmt}. Momentum=${momFmt}%.`,
    };
  }

  // ───────────────────────────────────────────────────────────────────────────
  // STRATEGY 4: KELLY CRITERION + VARIANCE RATIO (Evolved parameters)
  // ───────────────────────────────────────────────────────────────────────────
  getConservativeSignal(
    indicators: TechnicalIndicators,
    priceHistory: number[] = [],
    tradeStats: { winRate: number; avgWin: number; avgLoss: number } = { winRate: 0.5, avgWin: 0.01, avgLoss: 0.008 },
    genes?: MathGenes
  ): StrategySignal {
    const prices = priceHistory.length > 0 ? priceHistory : [indicators.currentPrice];

    const kellyFrac = genes ? genes.kellyFraction : 0.25;
    const kelly = kellyCriterion(tradeStats.winRate, tradeStats.avgWin, tradeStats.avgLoss, kellyFrac);

    const longPeriod = genes ? genes.varianceRatioLongPeriod : 10;
    const vr = varianceRatio(prices, 2, longPeriod);

    const z = zScore(prices, 20);

    const noiseRatio = genes ? genes.kalmanNoiseRatio : 0.005;
    const kalman = kalmanFilter(prices.slice(-30), noiseRatio);
    const { velocity } = kalman;

    const H = hurstExponent(prices.slice(-30));

    const kellyFmt = (kelly * 100).toFixed(1);
    const vrFmt = vr.toFixed(3);
    const HFmt = H.toFixed(3);
    const zFmt = z.toFixed(2);

    if (kelly <= 0) {
      return {
        action: 'HOLD',
        confidence: 0.5,
        reason: `Kelly Conservative: NEGATIVE EV (f*=${kellyFmt}%). Expected value negative, skipping. Payoff ratio=${(tradeStats.avgWin / (tradeStats.avgLoss || 1)).toFixed(2)}.`,
      };
    }

    const vrMomentum = vr > 1.05;

    if (z < -1.5 && velocity > 0 && kelly > 0.02) {
      const confidence = Math.min(0.92, 0.72 + kelly * 2 + Math.abs(z) * 0.04);
      return {
        action: 'BUY',
        confidence,
        reason: `Kelly Conservative BUY: f*=+${kellyFmt}% EV (positive). Z=${zFmt}σ. Evolved Kelly Sizing Factor: ${kellyFrac.toFixed(2)}.`,
      };
    }

    if (z > 1.5 && velocity < 0 && kelly > 0.02) {
      const confidence = Math.min(0.91, 0.72 + kelly * 2 + Math.abs(z) * 0.04);
      return {
        action: 'SELL',
        confidence,
        reason: `Kelly Conservative SELL: f*=+${kellyFmt}% EV (short-side). Z=+${zFmt}σ. Evolved Kelly Sizing Factor: ${kellyFrac.toFixed(2)}.`,
      };
    }

    if (vrMomentum && H > 0.55 && velocity > 0 && z > -0.5 && z < 1.0 && kelly > 0.05) {
      return {
        action: 'BUY',
        confidence: 0.80,
        reason: `Kelly Conservative: momentum regime VR=${vrFmt}, H=${HFmt}. Evolved Kelly sizing applied.`,
      };
    }

    return {
      action: 'HOLD',
      confidence: 0.5,
      reason: `Kelly Conservative: positive edge f*=+${kellyFmt}% but entry timing not optimal.`,
    };
  }

  // ───────────────────────────────────────────────────────────────────────────
  // UNIFIED SUPER STRATEGY — Democratic Weighted Voting across all 4 models
  // Each strategy votes BUY or SELL with its confidence. HOLD = abstain.
  // Final decision = highest weighted vote score above minimum quorum.
  // ───────────────────────────────────────────────────────────────────────────
  getUnifiedSignal(
    indicators: TechnicalIndicators,
    priceHistory: number[] = [],
    hasActiveTrade: boolean = false,
    averageEntryPrice: number = 0,
    tradeStats: { winRate: number; avgWin: number; avgLoss: number } = { winRate: 0.5, avgWin: 0.01, avgLoss: 0.008 },
    genes?: MathGenes,
    gemmaSignal?: { action: 'BUY' | 'SELL' | 'HOLD'; confidence: number; reason: string } | null
  ): StrategySignal {
    // Run all 4 strategies
    const oracle = this.getTemporalOracleSignal(indicators, [], priceHistory, genes);
    const statArb = this.getGridDcaSignal(indicators, hasActiveTrade, averageEntryPrice, priceHistory, genes);
    const kalmanHurst = this.getAiNeuralNetSignal(indicators, priceHistory, genes);
    const kelly = this.getConservativeSignal(indicators, priceHistory, tradeStats, genes);

    const votes = [oracle, statArb, kalmanHurst, kelly];
    const names = ['Binomial Oracle', 'Z-Score StatArb', 'Kalman+Hurst', 'Kelly Criterion'];

    if (gemmaSignal) {
      votes.push(gemmaSignal);
      names.push('Gemma 4 Crecetrader');
    }

    // Weighted vote tally
    let buyScore = 0;
    let sellScore = 0;
    const buyVoters: string[] = [];
    const sellVoters: string[] = [];
    const holdVoters: string[] = [];

    votes.forEach((v, i) => {
      if (v.action === 'BUY') {
        buyScore += v.confidence;
        buyVoters.push(`${names[i]}(${(v.confidence * 100).toFixed(0)}%)`);
      } else if (v.action === 'SELL') {
        sellScore += v.confidence;
        sellVoters.push(`${names[i]}(${(v.confidence * 100).toFixed(0)}%)`);
      } else {
        holdVoters.push(names[i]);
      }
    });

    // Minimum quorum: at least 1 strategy must vote + net score must exceed threshold
    const MIN_QUORUM_SCORE = 0.70;

    if (buyScore > sellScore && buyScore >= MIN_QUORUM_SCORE) {
      const avgConfidence = buyScore / Math.max(1, buyVoters.length);
      return {
        action: 'BUY',
        confidence: Math.min(0.98, avgConfidence),
        reason: `🗳️ UNIFIED VOTE BUY [${buyVoters.join(', ')}] | SELL:[${sellVoters.join(', ') || 'none'}] | Score: ${buyScore.toFixed(2)} vs ${sellScore.toFixed(2)}`,
      };
    }

    if (sellScore > buyScore && sellScore >= MIN_QUORUM_SCORE) {
      const avgConfidence = sellScore / Math.max(1, sellVoters.length);
      return {
        action: 'SELL',
        confidence: Math.min(0.98, avgConfidence),
        reason: `🗳️ UNIFIED VOTE SELL [${sellVoters.join(', ')}] | BUY:[${buyVoters.join(', ') || 'none'}] | Score: ${sellScore.toFixed(2)} vs ${buyScore.toFixed(2)}`,
      };
    }

    return {
      action: 'HOLD',
      confidence: 0.5,
      reason: `🗳️ UNIFIED VOTE: No quorum. BUY(${buyScore.toFixed(2)}) vs SELL(${sellScore.toFixed(2)}). Abstaining: [${holdVoters.join(', ')}]`,
    };
  }
}
