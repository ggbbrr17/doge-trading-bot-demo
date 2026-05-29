import { CustomNeuralNetwork } from './aiModel';

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
 *
 * Used by: Renaissance Technologies, DE Shaw, quantitative hedge funds.
 * Reference: Hurst (1951), Peters (1994) "Fractal Market Analysis"
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
 * Optimal linear estimator for noisy price observations.
 * Separates "true price signal" from market microstructure noise.
 *
 * Used by: Citadel, Two Sigma, quant trading desks.
 * Reference: Kalman (1960) IEEE Transactions on Automatic Control
 *
 * Q = process noise (how fast the true price can move)
 * R = measurement noise (how noisy the observed price is)
 */
function kalmanFilter(prices: number[]): { filtered: number[]; velocity: number; acceleration: number } {
  const Q = 0.0001; // Low process noise → smoother filter
  const R_noise = 0.005; // Measurement noise

  let x = prices[0];
  let P = 1.0;
  const filtered: number[] = [x];

  for (let i = 1; i < prices.length; i++) {
    // PREDICT step
    P = P + Q;
    // UPDATE step
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
 * z = (price - μ) / σ
 * The backbone of statistical arbitrage ("stat arb").
 *
 * z > +2.0 → price is statistically overbought → SELL
 * z < -2.0 → price is statistically oversold  → BUY
 * |z| < 0.5 → price is near equilibrium → HOLD
 *
 * Used by: Two Sigma, Citadel, AQR Capital, D.E. Shaw
 * Reference: Gatev et al. (2006) "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"
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
 * f* = (b·p - q) / b = (b·p - (1-p)) / b
 * where:
 *   b = net odds (avg_win / avg_loss ratio)
 *   p = empirical win probability
 *   q = 1 - p (loss probability)
 *
 * f* > 0 → positive expected value → worth trading
 * f* < 0 → negative expected value → do NOT trade
 *
 * Used by: Ed Thorp (Beat the Dealer, Beat the Market),
 *          Renaissance Technologies, Medallion Fund.
 * Reference: Kelly (1956) Bell System Technical Journal
 */
function kellyCriterion(winRate: number, avgWin: number, avgLoss: number): number {
  if (avgLoss <= 0 || winRate <= 0 || winRate >= 1) return 0;
  const b = avgWin / Math.abs(avgLoss);
  const p = winRate;
  const q = 1 - p;
  const kelly = (b * p - q) / b;
  // Use fractional Kelly (25%) to reduce variance (common practice among professionals)
  return Math.max(-1, Math.min(1, kelly * 0.25));
}

/**
 * VARIANCE RATIO TEST (Lo & MacKinlay, 1988)
 * Tests whether price changes follow a random walk.
 *
 * VR > 1.1 → positive serial autocorrelation (momentum/trending)
 * VR < 0.9 → negative serial autocorrelation (mean-reverting)
 * VR ≈ 1.0 → random walk
 *
 * Reference: Lo & MacKinlay (1988) "Stock Market Prices Do Not Follow Random Walks"
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
 * Annualized price return over a lookback, skipping the last tick
 * to avoid short-term reversal contamination.
 *
 * Positive → upward momentum → BUY
 * Negative → downward momentum → SELL
 *
 * Reference: Jegadeesh & Titman (1993) Journal of Finance
 */
function momentumScore(prices: number[], lookback: number = 20, skip: number = 1): number {
  if (prices.length < lookback + skip + 1) return 0;
  const start = prices[prices.length - lookback - skip];
  const end = prices[prices.length - 1 - skip];
  if (start === 0) return 0;
  return (end - start) / start;
}

// ═══════════════════════════════════════════════════════════════════════════════
// KEEP calculateIndicators for UI compatibility and Neural Network inputs
// (Bollinger Bands remain for price band visualization in the dashboard chart)
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
  // STRATEGY 1: BINOMIAL ORACLE
  // Uses the Binomial Distribution to estimate whether the current streak of
  // up/down moves is statistically expected or anomalous.
  //
  // Logic:
  //   - Count up-moves in the last N ticks
  //   - Assume fair-coin hypothesis (p = 0.5 baseline, adjusted by momentum)
  //   - P(X >= k | n, p): if the streak is statistically improbable in one
  //     direction, expect mean reversion (contrarian) or momentum (trend)
  //   - Combine with Kalman velocity for direction confirmation
  // ───────────────────────────────────────────────────────────────────────────
  getTemporalOracleSignal(
    indicators: TechnicalIndicators,
    futurePrices: number[],
    priceHistory: number[] = []
  ): StrategySignal {
    const prices = priceHistory.length > 0 ? priceHistory : [indicators.currentPrice];
    const N = Math.min(20, prices.length - 1);
    if (N < 5) {
      return { action: 'HOLD', confidence: 0.5, reason: 'Binomial Oracle: insufficient price history for analysis.' };
    }

    const recentPrices = prices.slice(-N - 1);
    const { upMoves, total } = countUpMoves(recentPrices);
    const downMoves = total - upMoves;

    // Base probability: use recent momentum to adjust p slightly
    const momentum = momentumScore(prices, Math.min(30, prices.length - 1));
    // Drift-adjusted p: if momentum is positive, slightly above 0.5
    const p = Math.max(0.2, Math.min(0.8, 0.5 + momentum * 5));

    // Kalman velocity for direction confirmation
    const kalman = kalmanFilter(recentPrices);
    const velocity = kalman.velocity;
    const acceleration = kalman.acceleration;

    // BINOMIAL ANALYSIS
    // P(X >= upMoves) under null hypothesis (p=0.5 fair coin)
    const pUpTail = binomialTailProbability(total, upMoves, 0.5);
    // P(X >= downMoves) for the sell direction
    const pDownTail = binomialTailProbability(total, downMoves, 0.5);

    const upPct = ((upMoves / total) * 100).toFixed(0);
    const pUpFmt = (pUpTail * 100).toFixed(1);
    const pDownFmt = (pDownTail * 100).toFixed(1);

    // BUY: upward trend is statistically SUPPORTED (probable to continue)
    // AND Kalman filter confirms positive velocity
    if (pUpTail > 0.70 && velocity > 0 && acceleration >= 0 && upMoves > downMoves) {
      const confidence = Math.min(0.97, 0.7 + (pUpTail - 0.7) * 0.9);
      return {
        action: 'BUY',
        confidence,
        reason: `Binomial Oracle: ${upMoves}/${total} up-ticks (${upPct}%). P(run|H₀)=${pUpFmt}% — uptrend statistically probable. Kalman velocity: +${(velocity * 10000).toFixed(4)}. Momentum bias: ${(momentum * 100).toFixed(2)}%.`,
      };
    }

    // SELL: upward streak is statistically EXHAUSTED (improbable to continue)
    // OR downward momentum is statistically confirmed
    if ((pUpTail < 0.15 && upMoves > downMoves) || (pDownTail > 0.75 && velocity < 0)) {
      const confidence = Math.min(0.96, 0.72 + (1 - pUpTail) * 0.25);
      return {
        action: 'SELL',
        confidence,
        reason: `Binomial Oracle: streak exhaustion detected. P(${upMoves} up in ${total}|H₀)=${pUpFmt}% — statistically improbable continuation. Kalman velocity: ${(velocity * 10000).toFixed(4)}. Mean reversion expected.`,
      };
    }

    // SELL: downward run is statistically dominant
    if (pDownTail > 0.80 && velocity < 0 && downMoves > upMoves) {
      const confidence = Math.min(0.95, 0.70 + (pDownTail - 0.70) * 0.85);
      return {
        action: 'SELL',
        confidence,
        reason: `Binomial Oracle: bearish dominance confirmed. ${downMoves}/${total} down-ticks. P(down run|H₀)=${pDownFmt}%. Kalman acceleration: ${(acceleration * 10000).toFixed(4)}.`,
      };
    }

    return {
      action: 'HOLD',
      confidence: 0.5,
      reason: `Binomial Oracle: no statistically significant edge. Up=${upMoves}/${total} (P=${pUpFmt}%). Market in equilibrium zone.`,
    };
  }

  // ───────────────────────────────────────────────────────────────────────────
  // STRATEGY 2: Z-SCORE STATISTICAL ARBITRAGE (DCA/GRID mode)
  // Mean-reversion strategy based on statistical deviation of price from its
  // rolling mean. Foundation of Statistical Arbitrage used by Two Sigma,
  // Citadel, and AQR Capital.
  //
  // Entry: z < -2.0 (price is 2 standard deviations below mean → BUY)
  // Exit:  z > +1.5 (price reverts back toward mean → SELL)
  // DCA:   z < -3.0 (extreme dislocation → add to position)
  // ───────────────────────────────────────────────────────────────────────────
  getGridDcaSignal(
    indicators: TechnicalIndicators,
    hasActiveTrade: boolean,
    averageEntryPrice: number,
    priceHistory: number[] = []
  ): StrategySignal {
    const prices = priceHistory.length > 0 ? priceHistory : [indicators.currentPrice];
    const z30 = zScore(prices, 30);  // 30-tick Z-score
    const z15 = zScore(prices, 15);  // 15-tick Z-score (faster signal)
    const currentPrice = indicators.currentPrice;

    // Variance Ratio to determine if market is mean-reverting or trending
    const vr = varianceRatio(prices, 2, 10);
    // VR < 0.9 → confirmed mean-reverting regime (ideal for stat arb)
    const isMeanReverting = vr < 0.92;
    const vrFmt = vr.toFixed(3);
    const z30Fmt = z30.toFixed(2);
    const z15Fmt = z15.toFixed(2);

    if (!hasActiveTrade) {
      // Strong oversold dislocation: Z < -2σ in mean-reverting market
      if (z30 < -2.0 && z15 < -1.5 && isMeanReverting) {
        const confidence = Math.min(0.93, 0.75 + Math.abs(z30 + 2) * 0.06);
        return {
          action: 'BUY',
          confidence,
          reason: `Z-Score Stat Arb: price is ${Math.abs(z30).toFixed(2)}σ below 30-tick mean (z₃₀=${z30Fmt}, z₁₅=${z15Fmt}). Variance Ratio=${vrFmt} confirms mean-reverting regime. Statistical edge for reversion entry.`,
        };
      }
      // Moderate oversold in ANY regime
      if (z30 < -2.5) {
        return {
          action: 'BUY',
          confidence: 0.80,
          reason: `Z-Score entry: extreme statistical dislocation z=${z30Fmt}σ. Price likely to revert to mean. VR=${vrFmt}.`,
        };
      }
    } else {
      // DCA SAFETY ORDER: extreme further dislocation
      const dropPct = ((averageEntryPrice - currentPrice) / averageEntryPrice) * 100;
      if (z30 < -3.0 && dropPct > 1.5) {
        return {
          action: 'BUY',
          confidence: 0.88,
          reason: `Z-Score DCA: extreme dislocation z=${z30Fmt}σ and price down ${dropPct.toFixed(2)}% from entry. Averaging down at statistically extreme level.`,
        };
      }

      // EXIT: price has reverted toward mean (z > +1.0 from entry)
      const profitPct = ((currentPrice - averageEntryPrice) / averageEntryPrice) * 100;
      if (z30 > 1.5 || (z30 > 0.8 && profitPct > 0.5)) {
        const confidence = Math.min(0.92, 0.75 + (z30 - 1.0) * 0.08);
        return {
          action: 'SELL',
          confidence,
          reason: `Z-Score exit: price reverted to z=${z30Fmt}σ from mean. PnL: +${profitPct.toFixed(2)}%. Statistical edge exhausted, closing position.`,
        };
      }

      // Overbought exit at extreme upper bound
      if (z30 > 2.5) {
        return {
          action: 'SELL',
          confidence: 0.90,
          reason: `Z-Score overbought: z=${z30Fmt}σ above mean. Sell signal — statistical reversion expected.`,
        };
      }
    }

    return {
      action: 'HOLD',
      confidence: 0.5,
      reason: `Z-Score equilibrium: z₃₀=${z30Fmt}σ, z₁₅=${z15Fmt}σ. VR=${vrFmt} (${isMeanReverting ? 'mean-reverting' : 'trending'} regime). No edge detected.`,
    };
  }

  // ───────────────────────────────────────────────────────────────────────────
  // STRATEGY 3: KALMAN FILTER + HURST EXPONENT (replaces Neural Network)
  // Regime-adaptive quantitative strategy combining:
  //   - Kalman Filter: optimal price signal extraction
  //   - Hurst Exponent: market regime classification
  //   - Momentum Factor: Jegadeesh & Titman academic factor
  //
  // Regime detection:
  //   H > 0.55 → TRENDING → follow Kalman velocity (momentum)
  //   H < 0.45 → MEAN-REVERTING → fade Kalman velocity (contrarian)
  //   H ≈ 0.50 → RANDOM WALK → reduce position size, wait
  // ───────────────────────────────────────────────────────────────────────────
  getAiNeuralNetSignal(indicators: TechnicalIndicators, priceHistory: number[] = []): StrategySignal {
    const prices = priceHistory.length > 0 ? priceHistory : [indicators.currentPrice];
    if (prices.length < 15) {
      return { action: 'HOLD', confidence: 0.5, reason: 'Kalman/Hurst: insufficient data.' };
    }

    // KALMAN FILTER: extract clean price signal and velocity
    const kalman = kalmanFilter(prices.slice(-50));
    const { velocity, acceleration } = kalman;

    // HURST EXPONENT: classify market regime
    const H = hurstExponent(prices.slice(-40));
    const isTrending = H > 0.55;
    const isMeanReverting = H < 0.45;

    // MOMENTUM FACTOR (Jegadeesh & Titman)
    const mom20 = momentumScore(prices, Math.min(20, prices.length - 2));
    const mom10 = momentumScore(prices, Math.min(10, prices.length - 2));

    // Z-score for mean-reversion component
    const z = zScore(prices, Math.min(25, prices.length));

    const HFmt = H.toFixed(3);
    const velFmt = (velocity * 10000).toFixed(4);
    const accFmt = (acceleration * 10000).toFixed(4);
    const momFmt = (mom20 * 100).toFixed(3);
    const regime = isTrending ? 'TRENDING' : isMeanReverting ? 'MEAN-REVERTING' : 'RANDOM WALK';

    if (isTrending) {
      // TRENDING REGIME → follow Kalman velocity + momentum confirmation
      if (velocity > 0 && acceleration >= 0 && mom20 > 0.001 && mom10 > 0) {
        const confidence = Math.min(0.94, 0.72 + H * 0.25 + mom20 * 5);
        return {
          action: 'BUY',
          confidence,
          reason: `Kalman+Hurst [${regime}]: H=${HFmt} confirms persistence. Kalman velocity=+${velFmt}, acceleration=+${accFmt}. Momentum (20)=${momFmt}% positive. Riding trending regime.`,
        };
      }
      if (velocity < 0 && acceleration <= 0 && mom20 < -0.001) {
        const confidence = Math.min(0.93, 0.72 + H * 0.25 + Math.abs(mom20) * 5);
        return {
          action: 'SELL',
          confidence,
          reason: `Kalman+Hurst [${regime}]: H=${HFmt} downtrend persistent. Kalman velocity=${velFmt}, momentum=${momFmt}%. Exiting trending bearish regime.`,
        };
      }
    } else if (isMeanReverting) {
      // MEAN-REVERTING REGIME → fade extremes (contrarian)
      if (z < -1.8 && velocity > 0) {
        // Price is below mean but Kalman shows recovery starting
        const confidence = Math.min(0.91, 0.73 + (1 - H) * 0.2);
        return {
          action: 'BUY',
          confidence,
          reason: `Kalman+Hurst [${regime}]: H=${HFmt} anti-persistent. Z-score=${z.toFixed(2)}σ oversold + Kalman recovery velocity=+${velFmt}. Contrarian entry.`,
        };
      }
      if (z > 1.8 && velocity < 0) {
        const confidence = Math.min(0.90, 0.73 + (1 - H) * 0.2);
        return {
          action: 'SELL',
          confidence,
          reason: `Kalman+Hurst [${regime}]: H=${HFmt} anti-persistent. Z-score=+${z.toFixed(2)}σ overbought + Kalman decline velocity=${velFmt}. Contrarian exit.`,
        };
      }
    }

    return {
      action: 'HOLD',
      confidence: 0.5,
      reason: `Kalman+Hurst [${regime}]: H=${HFmt}. Kalman vel=${velFmt}, acc=${accFmt}. Momentum=${momFmt}%. No asymmetric edge found.`,
    };
  }

  // ───────────────────────────────────────────────────────────────────────────
  // STRATEGY 4: KELLY CRITERION + VARIANCE RATIO (Conservative)
  // Position sizing meets regime detection.
  // Only enters when the KELLY FRACTION is positive (positive expected value)
  // AND the VARIANCE RATIO confirms the market regime matches strategy type.
  //
  // Kelly Criterion is used by:
  //   - Ed Thorp (Pioneer of quantitative trading, Black-Scholes precursor)
  //   - Renaissance Technologies Medallion Fund
  //   - Warren Buffett (qualitative version)
  //
  // Uses historical closed-trade stats to compute win rate and payoff ratio.
  // ───────────────────────────────────────────────────────────────────────────
  getConservativeSignal(
    indicators: TechnicalIndicators,
    priceHistory: number[] = [],
    tradeStats: { winRate: number; avgWin: number; avgLoss: number } = { winRate: 0.5, avgWin: 0.01, avgLoss: 0.008 }
  ): StrategySignal {
    const prices = priceHistory.length > 0 ? priceHistory : [indicators.currentPrice];
    const currentPrice = indicators.currentPrice;

    // KELLY FRACTION: mathematical edge measurement
    const kelly = kellyCriterion(tradeStats.winRate, tradeStats.avgWin, tradeStats.avgLoss);

    // VARIANCE RATIO for regime confirmation
    const vr = varianceRatio(prices, 2, 10);

    // Z-SCORE for entry timing
    const z = zScore(prices, 20);

    // KALMAN for direction
    const kalman = kalmanFilter(prices.slice(-30));
    const { velocity } = kalman;

    // HURST for regime
    const H = hurstExponent(prices.slice(-30));

    const kellyFmt = (kelly * 100).toFixed(1);
    const vrFmt = vr.toFixed(3);
    const HFmt = H.toFixed(3);
    const zFmt = z.toFixed(2);

    // Only trade if Kelly fraction is POSITIVE (positive expected value)
    if (kelly <= 0) {
      return {
        action: 'HOLD',
        confidence: 0.5,
        reason: `Kelly Conservative: NEGATIVE edge (f*=${kellyFmt}%). Trade has negative expected value — standing aside. Win rate=${(tradeStats.winRate * 100).toFixed(0)}%, Payoff ratio=${(tradeStats.avgWin / (tradeStats.avgLoss || 1)).toFixed(2)}.`,
      };
    }

    // Positive Kelly edge confirmed — now time the entry with Z-score + Kalman
    const vrMomentum = vr > 1.05;  // Variance ratio suggests momentum
    const vrReversion = vr < 0.95; // Variance ratio suggests mean reversion

    // BUY CONDITION: positive Kelly + price is statistically oversold + Kalman up
    if (z < -1.5 && velocity > 0 && kelly > 0.02) {
      const confidence = Math.min(0.92, 0.72 + kelly * 2 + Math.abs(z) * 0.04);
      return {
        action: 'BUY',
        confidence,
        reason: `Kelly Conservative BUY: f*=+${kellyFmt}% (positive edge). Z=${zFmt}σ oversold, Kalman recovering (vel=+${(velocity * 10000).toFixed(3)}). H=${HFmt}, VR=${vrFmt}. Mathematically favorable entry.`,
      };
    }

    // SELL CONDITION: positive Kelly but price overbought + Kalman declining
    if (z > 1.5 && velocity < 0 && kelly > 0.02) {
      const confidence = Math.min(0.91, 0.72 + kelly * 2 + Math.abs(z) * 0.04);
      return {
        action: 'SELL',
        confidence,
        reason: `Kelly Conservative SELL: f*=+${kellyFmt}% edge on short side. Z=+${zFmt}σ overbought, Kalman declining. H=${HFmt}, VR=${vrFmt}. Exiting statistically extended position.`,
      };
    }

    // Strong momentum buy in trending regime with positive Kelly
    if (vrMomentum && H > 0.55 && velocity > 0 && z > -0.5 && z < 1.0 && kelly > 0.05) {
      return {
        action: 'BUY',
        confidence: 0.80,
        reason: `Kelly Conservative: momentum regime confirmed. f*=+${kellyFmt}%, VR=${vrFmt} (trending), H=${HFmt}. Entering with mathematically sized position.`,
      };
    }

    return {
      action: 'HOLD',
      confidence: 0.5,
      reason: `Kelly Conservative: f*=+${kellyFmt}% (positive edge but timing not optimal). Z=${zFmt}σ, VR=${vrFmt}, H=${HFmt}. Waiting for better entry point.`,
    };
  }
}
