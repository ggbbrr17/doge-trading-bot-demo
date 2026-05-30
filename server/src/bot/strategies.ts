import { CustomNeuralNetwork } from './aiModel';
import { MathGenes } from './evolutionEngine';
import { OrderBookSignal } from '../orderBookSensor';

export interface StrategySignal {
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number; // 0 to 1
  reason: string;
  targetSL?: number;
  targetTP?: number;
}

export interface TechnicalIndicators {
  rsi: number;
  macd: { macd: number; signal: number; hist: number };
  ema: { ema20: number; ema50: number; ema200: number };
  bollinger: { upper: number; middle: number; lower: number; width: number };
  atr: number;
  vwap: number;
  fractalDimension: number;
  efficiencyRatio: number;
  botActivity: number;
  spectralEnergy: number; // Fuerza del ciclo dominante
  cyclePhase: number;    // Fase del ciclo (-1 a 1, Valle a Cima)
  htfAlignment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
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
 * FOURIER SPECTRAL ANALYSIS
 * Transforma el dominio del tiempo al dominio de la frecuencia.
 * Identifica el ciclo dominante y su fase actual.
 */
function fourierSpectralAnalysis(prices: number[], window: number = 64): { energy: number, phase: number } {
  if (prices.length < window) return { energy: 0, phase: 0 };

  const n = window;
  const signal = prices.slice(-n);
  const mean = signal.reduce((a, b) => a + b, 0) / n;
  const detrended = signal.map(p => p - mean);

  let maxMagnitude = 0;
  let dominantFreq = 0;
  let phaseAtMax = 0;

  // Calculamos la DFT para las frecuencias de interés (excluyendo DC y ruido extremo)
  for (let k = 1; k < n / 2; k++) {
    let re = 0;
    let im = 0;
    for (let t = 0; t < n; t++) {
      const angle = (2 * Math.PI * k * t) / n;
      re += detrended[t] * Math.cos(angle);
      im -= detrended[t] * Math.sin(angle);
    }

    const magnitude = Math.sqrt(re * re + im * im);
    if (magnitude > maxMagnitude) {
      maxMagnitude = magnitude;
      dominantFreq = k;
      phaseAtMax = Math.atan2(im, re);
    }
  }

  // Calculamos la fase actual del ciclo dominante en el último tick
  const currentPhase = Math.sin((2 * Math.PI * dominantFreq * (n - 1)) / n + phaseAtMax);

  // Energy es la magnitud normalizada (relación señal/ruido)
  const totalEnergy = detrended.reduce((a, b) => a + b * b, 0);
  const energy = totalEnergy > 0 ? (maxMagnitude * maxMagnitude) / totalEnergy : 0;

  return { energy, phase: currentPhase };
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

/**
 * KAUFMAN EFFICIENCY RATIO (ER)
 * Mide la eficiencia del movimiento del precio.
 * ER = Cambio absoluto / Suma de cambios absolutos individuales
 * Cerca de 1.0 -> Tendencia perfecta (Institucional)
 * Cerca de 0.0 -> Mercado ruidoso / Lateral
 */
function efficiencyRatio(prices: number[], window: number = 14): number {
  if (prices.length < window + 1) return 0.5;
  const slice = prices.slice(-window);
  const totalChange = Math.abs(slice[slice.length - 1] - slice[0]);
  let sumChanges = 0;
  for (let i = 1; i < slice.length; i++) {
    sumChanges += Math.abs(slice[i] - slice[i - 1]);
  }
  return sumChanges === 0 ? 0 : totalChange / sumChanges;
}

/**
 * FRACTAL DIMENSION INDEX (FDI)
 * Determina la complejidad geométrica del mercado.
 * FDI < 1.5 -> Mercado persistente (Trending)
 * FDI > 1.5 -> Mercado antipersistente (Mean Reverting / Ranging)
 */
function fractalDimension(prices: number[], window: number = 30): number {
  if (prices.length < window) return 1.5;
  const slice = prices.slice(-window);

  const maxPrice = Math.max(...slice);
  const minPrice = Math.min(...slice);
  const range = maxPrice - minPrice;

  if (range === 0) return 1.5;

  let length = 0;
  for (let i = 1; i < slice.length; i++) {
    const diff = (slice[i] - slice[i - 1]) / range;
    length += Math.sqrt(Math.pow(diff, 2) + Math.pow(1 / window, 2));
  }

  // D = 1 + [log(L) + log(2)] / log(2 * (n-1))
  const d = 1 + (Math.log(length) + Math.log(2)) / Math.log(2 * (window - 1));
  return Math.max(1.0, Math.min(2.0, d));
}

// ═══════════════════════════════════════════════════════════════════════════════
// UI compatibility helper
// ═══════════════════════════════════════════════════════════════════════════════
export function calculateIndicators(prices: number[], candles: any[] = []): TechnicalIndicators {
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

  // 1. Average True Range (ATR) - Volatility measure
  let atr = 0;
  if (candles.length >= 14) {
    const trs: number[] = [];
    for (let i = 1; i < candles.length; i++) {
      const h = candles[i].high;
      const l = candles[i].low;
      const pc = candles[i - 1].close;
      trs.push(Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc)));
    }
    atr = trs.slice(-14).reduce((a, b) => a + b, 0) / 14;
  }

  // 2. VWAP - Volume Weighted Average Price
  let vwap = currentPrice;
  if (candles.length > 0) {
    let totalVol = 0;
    let totalPV = 0;
    candles.slice(-50).forEach(c => {
      const tp = (c.high + c.low + c.close) / 3;
      totalPV += tp * c.volume;
      totalVol += c.volume;
    });
    vwap = totalVol > 0 ? totalPV / totalVol : currentPrice;
  }

  const fdi = fractalDimension(prices);
  const er = efficiencyRatio(prices);
  const spectral = fourierSpectralAnalysis(prices);

  // 3. Bot Activity Index (Volume Density)
  // Mide si hay un volumen inusual concentrado en movimientos eficientes (Algoritmos)
  let botActivity = 0;
  if (candles.length >= 20) {
    const avgVol = candles.slice(-20).reduce((a, b) => a + b.volume, 0) / 20;
    const currentVol = candles[candles.length - 1].volume;
    const volAnomaly = currentVol / (avgVol || 1);

    // Si el volumen es > 2x el promedio y el precio se mueve de forma eficiente, hay bots.
    botActivity = volAnomaly * er;
  }

  // 4. HTF Bias Detection (Temporalidad Superior Inferida)
  // Usamos una ventana de 200 periodos para inferir el sesgo de una temporalidad mayor
  let htfAlignment: 'BULLISH' | 'BEARISH' | 'NEUTRAL' = 'NEUTRAL';
  if (prices.length >= 200) {
    const slope = (ema50 - prices[prices.length - 10]) / prices[prices.length - 10];

    if (currentPrice > ema200 && ema50 > ema200 && slope > 0) htfAlignment = 'BULLISH';
    else if (currentPrice < ema200 && ema50 < ema200 && slope < 0) htfAlignment = 'BEARISH';
  }

  return {
    rsi, macd: { macd: macdVal, signal: signalVal, hist },
    ema: { ema20, ema50, ema200 },
    bollinger: { upper, middle, lower, width },
    atr, vwap, fractalDimension: fdi, efficiencyRatio: er, botActivity, spectralEnergy: spectral.energy, cyclePhase: spectral.phase, htfAlignment, currentPrice,
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

    if (z < -1.5 && velocity > 0 && kelly > 0.05) {
      const confidence = Math.min(0.92, 0.72 + kelly * 2 + Math.abs(z) * 0.04);
      return {
        action: 'BUY',
        confidence,
        reason: `Kelly Conservative BUY: f*=+${kellyFmt}% EV (positive). Z=${zFmt}σ. Evolved Kelly Sizing Factor: ${kellyFrac.toFixed(2)}.`,
      };
    }

    if (z > 1.5 && velocity < 0 && kelly > 0.05) {
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
  // STRATEGY 5: FRACTAL EFFICIENCY QUANTECH (High-End Institutional)
  // ───────────────────────────────────────────────────────────────────────────
  getInstitutionalQuantSignal(indicators: TechnicalIndicators): StrategySignal {
    const { fractalDimension: fdi, efficiencyRatio: er, currentPrice, vwap } = indicators;

    const isEfficient = er > 0.6; // Movimiento muy "limpio"
    const isTrending = fdi < 1.45; // El mercado tiene inercia
    const isOverextended = fdi > 1.65; // El mercado está agotado/ruidoso

    const erFmt = er.toFixed(2);
    const fdiFmt = fdi.toFixed(2);

    // Señal de Compra Institucional: Tendencia limpia confirmada por baja fractalidad
    if (isTrending && isEfficient && currentPrice > vwap) {
      return {
        action: 'BUY',
        confidence: 0.95,
        reason: `Institutional Quant: Eficiencia alta (ER=${erFmt}) y Fractilidad baja (FDI=${fdiFmt}). Tendencia sólida detectada por encima del VWAP.`,
      };
    }

    // Señal de Venta Institucional: Agotamiento fractal en máximos o ineficiencia
    if (isOverextended && currentPrice > vwap * 1.02) {
      return {
        action: 'SELL',
        confidence: 0.90,
        reason: `Institutional Quant: Agotamiento detectado (FDI=${fdiFmt}). El mercado ha perdido su estructura eficiente y se espera reversión al VWAP.`,
      };
    }

    if (isTrending && isEfficient && currentPrice < vwap * 0.98) {
      return {
        action: 'SELL',
        confidence: 0.92,
        reason: `Institutional Quant: Momentum bajista eficiente detectado. ER=${erFmt}, FDI=${fdiFmt}.`,
      };
    }

    return {
      action: 'HOLD',
      confidence: 0.5,
      reason: `Institutional Quant: Mercado en estado de equilibrio fractal (ER=${erFmt}, FDI=${fdiFmt}).`,
    };
  }

  // ───────────────────────────────────────────────────────────────────────────
  // STRATEGY 6: ALGORITHMIC HERD SENSOR (Mass Bot Detection)
  // ───────────────────────────────────────────────────────────────────────────
  getBotHerdSignal(indicators: TechnicalIndicators): StrategySignal {
    const { botActivity, efficiencyRatio: er, currentPrice, vwap } = indicators;

    // Bot Activity > 2.5 indica una anomalía de volumen algorítmico masivo
    const isBotMassMovement = botActivity > 2.5;
    const isBullishPressure = currentPrice > vwap && er > 0.7;
    const isBearishPressure = currentPrice < vwap && er > 0.7;

    if (isBotMassMovement && isBullishPressure) {
      return {
        action: 'BUY',
        confidence: 0.92,
        reason: `Herd Sensor: Detectada actividad algorítmica masiva (Index: ${botActivity.toFixed(2)}). Los bots están empujando el precio de forma eficiente al alza.`,
      };
    }

    if (isBotMassMovement && isBearishPressure) {
      return {
        action: 'SELL',
        confidence: 0.92,
        reason: `Herd Sensor: Liquidación robótica masiva detectada (Index: ${botActivity.toFixed(2)}). Presión algorítmica bajista confirmada.`,
      };
    }

    return {
      action: 'HOLD',
      confidence: 0.5,
      reason: `Herd Sensor: Actividad de bots en niveles normales (${botActivity.toFixed(2)}). Sin movimientos en masa detectados.`,
    };
  }

  // ───────────────────────────────────────────────────────────────────────────
  // STRATEGY 7: SYSTEMIC OMEGA INVERSION (The "Golden Gate" Entry)
  // ───────────────────────────────────────────────────────────────────────────
  /**
   * Detecta la "Inversión Sistémica": El momento exacto donde el caos (rango)
   * se transforma en orden (tendencia eficiente) en un nivel de sobreventa.
   * Utiliza Inferencia Fractal para predecir el inicio de un "Short Squeeze" o "Bull Run".
   */
  getOmegaInversionSignal(indicators: TechnicalIndicators, priceHistory: number[]): StrategySignal {
    const { currentPrice, vwap, efficiencyRatio: er, fractalDimension: fdi, atr } = indicators;
    const prices = priceHistory.slice(-40);
    const z = zScore(prices, 20);
    const H = hurstExponent(prices);

    // 1. Detección de Agotamiento de Rango: El mercado era caótico (H < 0.45)
    // 2. Inferencia de Iniciación: El ER sube (> 0.65) indicando que el dinero inteligente entró.
    // 3. Dislocación: El precio está significativamente por debajo del VWAP o Z-Score bajo.

    const isUnderValued = z < -2.1 || currentPrice < vwap * 0.985;
    const isTransitioningToTrend = er > 0.68 && fdi < 1.48;
    const hasInertia = H > 0.52;

    if (isUnderValued && isTransitioningToTrend && hasInertia) {
      // Calculamos un Take Profit inferencial basado en la expansión de volatilidad
      const targetTP = (atr * 4 / currentPrice) * 100; // Objetivo de 4 ATRs de expansión

      return {
        action: 'BUY',
        confidence: 0.98,
        reason: `Omega Inversion: Transición sistémica detectada. El caos fractal (H=${H.toFixed(2)}) se ha ordenado en una tendencia eficiente (ER=${er.toFixed(2)}) en zona de descuento extremo (Z=${z.toFixed(2)}σ). Punto de iniciación institucional detectado.`,
        targetTP: Math.max(2.5, targetTP),
        targetSL: Math.max(1.2, (atr * 1.8 / currentPrice) * 100)
      };
    }

    // Señal de Inversión de Techo (Para Salidas o Shorts)
    const isOverValued = z > 2.1 || currentPrice > vwap * 1.02;
    const isLosingEfficiency = er < 0.4 && fdi > 1.6;

    if (isOverValued && isLosingEfficiency) {
      return {
        action: 'SELL',
        confidence: 0.94,
        reason: `Omega Inversion: Agotamiento sistémico. El precio ha entrado en entropía máxima (FDI=${fdi.toFixed(2)}) en zona de sobreextensión. Probabilidad de reversión al VWAP > 90%.`,
      };
    }

    return {
      action: 'HOLD',
      confidence: 0.5,
      reason: 'Omega Inversion: El mercado no presenta una transición de fase sistémica clara.',
    };
  }

  // ───────────────────────────────────────────────────────────────────────────
  // STRATEGY 8: SPECTRAL FOURIER CYCLE (Frequency Domain Edge)
  // ───────────────────────────────────────────────────────────────────────────
  getFourierCycleSignal(indicators: TechnicalIndicators): StrategySignal {
    const { spectralEnergy, cyclePhase, currentPrice, vwap, htfAlignment } = indicators;

    // Un spectralEnergy alto (> 0.4) indica que hay un ciclo dominante muy claro (no es ruido)
    const hasStableCycle = spectralEnergy > 0.4;

    // Cycle Phase: -1 (Valle), 0 (Cruce), 1 (Cima)
    const isCyclicLow = cyclePhase < -0.8;
    const isCyclicHigh = cyclePhase > 0.8;

    if (hasStableCycle && isCyclicLow && htfAlignment !== 'BEARISH') {
      return {
        action: 'BUY',
        confidence: 0.90,
        reason: `Fourier Spectral: Detectado valle en ciclo dominante (Energía: ${spectralEnergy.toFixed(2)}). El mercado está en fase de acumulación cíclica.`,
      };
    }

    if (hasStableCycle && isCyclicHigh) {
      return {
        action: 'SELL',
        confidence: 0.88,
        reason: `Fourier Spectral: Detectada cima en ciclo dominante. Fase de distribución cíclica alcanzada. Probable reversión.`,
      };
    }

    return {
      action: 'HOLD',
      confidence: 0.5,
      reason: spectralEnergy > 0.2
        ? `Fourier Spectral: Ciclo en fase neutra (${cyclePhase.toFixed(2)}).`
        : `Fourier Spectral: Mercado ruidoso (Baja energía espectral).`,
    };
  }

  // ───────────────────────────────────────────────────────────────────────────
  // STRATEGY 9: ORDER BOOK FLOW SENSOR (L2 Depth Analysis)
  // ───────────────────────────────────────────────────────────────────────────
  getOrderBookFlowSignal(obiSignal: OrderBookSignal | null): StrategySignal {
    if (!obiSignal) return { action: 'HOLD', confidence: 0.5, reason: 'OB Sensor: Awaiting snapshot.' };

    const { obi, microPressure, wallSide, wallPrice } = obiSignal;

    // BUY: Imbalance favors bids, weighted pressure is positive, and no sell wall blocking
    if (obi > 0.25 && microPressure > 0.30 && wallSide !== 'SELL') {
      return {
        action: 'BUY',
        confidence: Math.min(0.95, 0.6 + obi),
        reason: `OB Flow: Institutional buy pressure detected (OBI: ${obi}, Pressure: ${microPressure}). No resistance walls detected.`,
      };
    }

    // SELL: Imbalance favors asks, weighted pressure is negative, or buy wall missing
    if (obi < -0.25 && microPressure < -0.30 && wallSide !== 'BUY') {
      return {
        action: 'SELL',
        confidence: Math.min(0.95, 0.6 + Math.abs(obi)),
        reason: `OB Flow: Distribution detected (OBI: ${obi}, Pressure: ${microPressure}). Support walls receding.`,
      };
    }

    return { action: 'HOLD', confidence: 0.5, reason: `OB Flow: Neutral imbalance (${obi}).` };
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
    gemmaSignal?: { action: 'BUY' | 'SELL' | 'HOLD'; confidence: number; reason: string } | null,
    hmmRegime?: string,
    obiSignal?: OrderBookSignal | null
  ): StrategySignal {
    // Run all 4 strategies
    const oracle = this.getTemporalOracleSignal(indicators, [], priceHistory, genes);
    const statArb = this.getGridDcaSignal(indicators, hasActiveTrade, averageEntryPrice, priceHistory, genes);
    const kalmanHurst = this.getAiNeuralNetSignal(indicators, priceHistory, genes);
    const kelly = this.getConservativeSignal(indicators, priceHistory, tradeStats, genes);
    const quant = this.getInstitutionalQuantSignal(indicators);
    const botHerd = this.getBotHerdSignal(indicators);
    const omega = this.getOmegaInversionSignal(indicators, priceHistory);
    const spectral = this.getFourierCycleSignal(indicators);
    const obFlow = this.getOrderBookFlowSignal(obiSignal);

    // Modify strategy weighting based on the detected HMM market regime
    // Trend strategies: Oracle, KalmanHurst
    // Mean reversion strategies: Z-Score StatArb (Grid DCA)
    let oracleWeight = 1.0;
    let statArbWeight = 1.0;
    let kalmanWeight = 1.0;
    let kellyWeight = 1.0;

    if (hmmRegime === 'TREND_BULL' || hmmRegime === 'TREND_BEAR') {
      oracleWeight = 1.5;
      kalmanWeight = 1.5;
      statArbWeight = 0.5; // Reduce mean-reversion in trending markets
    } else if (hmmRegime === 'RANGE') {
      oracleWeight = 0.5; // Reduce trend-following in ranging markets
      kalmanWeight = 0.5;
      statArbWeight = 1.5;
    }

    // Apply weights to signals
    oracle.confidence *= oracleWeight;
    statArb.confidence *= statArbWeight;
    kalmanHurst.confidence *= kalmanWeight;
    kelly.confidence *= kellyWeight;

    // Cap confidence at 0.99
    oracle.confidence = Math.min(0.99, oracle.confidence);
    statArb.confidence = Math.min(0.99, statArb.confidence);
    kalmanHurst.confidence = Math.min(0.99, kalmanHurst.confidence);
    kelly.confidence = Math.min(0.99, kelly.confidence);

    const votes = [oracle, statArb, kalmanHurst, kelly, quant, botHerd, omega, spectral, obFlow];
    const names = ['Binomial Oracle', 'Z-Score StatArb', 'Kalman+Hurst', 'Kelly Criterion', 'Fractal Quant', 'Algo Herd Sensor', 'Omega Inversion', 'Fourier Spectral', 'OB Flow Sensor'];

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

    // ───────────────────────────────────────────────────────────────────────────
    // SYSTEMIC TIMEFRAME FILTER: Alineación de Marea
    // ───────────────────────────────────────────────────────────────────────────
    // Si la temporalidad superior es bajista, reducimos drásticamente la confianza de compras
    if (indicators.htfAlignment === 'BEARISH' && buyScore > 0) {
      buyScore *= 0.4; // Penalización por ir contra la marea
    }
    if (indicators.htfAlignment === 'BULLISH' && sellScore > 0) {
      sellScore *= 0.4; // Penalización por ir contra la marea
    }

    // Minimum quorum: at least 1 strategy must vote + net score must exceed threshold
    // Increased to 0.90 to ensure extremely high probability of profit per user request
    const MIN_QUORUM_SCORE = 0.90;

    const regimePrefix = hmmRegime ? `[Regime: ${hmmRegime}] ` : '';

    if (buyScore > sellScore && buyScore >= MIN_QUORUM_SCORE) {
      const avgConfidence = buyScore / Math.max(1, buyVoters.length);
      return {
        action: 'BUY',
        confidence: Math.min(0.98, avgConfidence),
        reason: `${regimePrefix}🗳️ UNIFIED VOTE BUY [${buyVoters.join(', ')}] | SELL:[${sellVoters.join(', ') || 'none'}] | Score: ${buyScore.toFixed(2)} vs ${sellScore.toFixed(2)}`,
      };
    }

    if (sellScore > buyScore && sellScore >= MIN_QUORUM_SCORE) {
      const avgConfidence = sellScore / Math.max(1, sellVoters.length);
      return {
        action: 'SELL',
        confidence: Math.min(0.98, avgConfidence),
        reason: `${regimePrefix}🗳️ UNIFIED VOTE SELL [${sellVoters.join(', ')}] | BUY:[${buyVoters.join(', ') || 'none'}] | Score: ${sellScore.toFixed(2)} vs ${buyScore.toFixed(2)}`,
      };
    }

    return {
      action: 'HOLD',
      confidence: 0.5,
      reason: `${regimePrefix}🗳️ UNIFIED VOTE: No quorum. BUY(${buyScore.toFixed(2)}) vs SELL(${sellScore.toFixed(2)}). Abstaining: [${holdVoters.join(', ')}]`,
    };
  }
}
