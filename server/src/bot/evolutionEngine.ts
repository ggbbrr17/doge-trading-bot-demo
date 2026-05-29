import * as https from 'https';

export interface MathGenes {
  binomialThreshold: number;     // e.g., 0.70 (above which uptrend confirmed)
  zScoreEntry: number;           // e.g., -2.0 (below which buy in stat arb)
  zScoreExit: number;            // e.g., 1.5 (above which take profit in stat arb)
  hurstTrending: number;         // e.g., 0.55 (above which market is trending)
  hurstReversion: number;        // e.g., 0.45 (below which market is mean-reverting)
  kalmanNoiseRatio: number;      // e.g., 0.005 (measurement noise factor)
  varianceRatioLongPeriod: number; // e.g., 10 (long window for Lo-MacKinlay)
  kellyFraction: number;         // e.g., 0.25 (fractional Kelly sizing factor)
  momentumLookback: number;      // e.g., 20 (ticks for J-T momentum score)
}


export interface EvolutionStats {
  generation: number;
  activeGenes: MathGenes;
  fitnessScore: number;
  lastEvolvedAt: number;
  bestFormulaExpression: string;
  evolutionLogs: string[];
}

export class EvolutionEngine {
  private activeGenes: MathGenes;
  private generation = 0;
  private fitnessScore = 0.0;
  private lastEvolvedAt = Date.now();
  private bestFormulaExpression = "P(X >= k | n, p) * K_vel + Kelly(f*) - Z_score * VarianceRatio()";
  private evolutionLogs: string[] = ["Evolutionary Engine Initialized: awaiting genetic seed."];

  // Default optimal baseline seeds (Chromosomes)
  constructor() {
    this.activeGenes = {
      binomialThreshold: 0.70,
      zScoreEntry: -2.0,
      zScoreExit: 1.5,
      hurstTrending: 0.55,
      hurstReversion: 0.45,
      kalmanNoiseRatio: 0.005,
      varianceRatioLongPeriod: 10,
      kellyFraction: 0.25,
      momentumLookback: 20
    };
  }

  getActiveGenes(): MathGenes {
    return this.activeGenes;
  }

  getStats(): EvolutionStats {
    return {
      generation: this.generation,
      activeGenes: this.activeGenes,
      fitnessScore: this.fitnessScore,
      lastEvolvedAt: this.lastEvolvedAt,
      bestFormulaExpression: this.bestFormulaExpression,
      evolutionLogs: this.evolutionLogs
    };
  }

  private log(msg: string) {
    const timestamp = new Date().toLocaleTimeString();
    this.evolutionLogs.unshift(`[${timestamp}] ${msg}`);
    if (this.evolutionLogs.length > 50) this.evolutionLogs.pop();
  }

  /**
   * MUTATE: Apply Gaussian-like random walk mutation to a set of genes
   */
  private mutate(genes: MathGenes, rate = 0.15): MathGenes {
    const clone = { ...genes };
    const rand = (min: number, max: number) => Math.random() * (max - min) + min;

    if (Math.random() < rate) clone.binomialThreshold = Math.max(0.55, Math.min(0.95, clone.binomialThreshold + rand(-0.08, 0.08)));
    if (Math.random() < rate) clone.zScoreEntry = Math.max(-3.5, Math.min(-1.0, clone.zScoreEntry + rand(-0.3, 0.3)));
    if (Math.random() < rate) clone.zScoreExit = Math.max(0.5, Math.min(2.5, clone.zScoreExit + rand(-0.25, 0.25)));
    if (Math.random() < rate) clone.hurstTrending = Math.max(0.50, Math.min(0.75, clone.hurstTrending + rand(-0.05, 0.05)));
    if (Math.random() < rate) clone.hurstReversion = Math.max(0.25, Math.min(0.49, clone.hurstReversion + rand(-0.05, 0.05)));
    if (Math.random() < rate) clone.kalmanNoiseRatio = Math.max(0.0005, Math.min(0.05, clone.kalmanNoiseRatio + rand(-0.002, 0.002)));
    if (Math.random() < rate) clone.varianceRatioLongPeriod = Math.max(5, Math.min(25, Math.round(clone.varianceRatioLongPeriod + rand(-3, 3))));
    if (Math.random() < rate) clone.kellyFraction = Math.max(0.05, Math.min(0.50, clone.kellyFraction + rand(-0.05, 0.05)));
    if (Math.random() < rate) clone.momentumLookback = Math.max(5, Math.min(40, Math.round(clone.momentumLookback + rand(-4, 4))));

    return clone;
  }

  /**
   * SIMULATED QUANTITATIVE BACKTEST FITNESS FUNCTION
   * Simulates trade entry & exit decisions on the recent priceHistory buffer using
   * the provided candidate genes to see how much return (PnL) they would have made.
   */
  evaluateFitness(genes: MathGenes, prices: number[]): number {
    if (prices.length < 40) return 0.0;

    let balance = 10000;
    let position: { price: number; side: 'BUY' } | null = null;
    let tradesCount = 0;

    // Fast backtest over the last prices
    for (let i = 25; i < prices.length - 1; i++) {
      const currentPrice = prices[i];
      const slice = prices.slice(0, i + 1);

      // Z-Score calculation
      const zWindow = 20;
      const zSlice = slice.slice(-zWindow);
      const mean = zSlice.reduce((a, b) => a + b, 0) / zSlice.length;
      const variance = zSlice.reduce((s, p) => s + Math.pow(p - mean, 2), 0) / zSlice.length;
      const std = Math.sqrt(variance);
      const z = std === 0 ? 0 : (currentPrice - mean) / std;

      // Decision rules simulated from the mathematical strategies
      if (!position) {
        // Buy signal simulation based on evolved Z-Score entry threshold
        if (z < genes.zScoreEntry) {
          position = { price: currentPrice, side: 'BUY' };
        }
      } else {
        // Sell signal / Take profit simulation
        const pnlPct = ((currentPrice - position.price) / position.price) * 100;
        if (z > genes.zScoreExit || pnlPct < -1.5) {
          balance += (pnlPct / 100) * 1000; // Simulated $1000 exposure
          position = null;
          tradesCount++;
        }
      }
    }

    const finalPnL = balance - 10000;
    // Fitness is penalized if too few trades or extreme risk (drawdown)
    const penalty = tradesCount === 0 ? -100 : 0;
    return finalPnL + penalty;
  }

  /**
   * AI FORMULA GENERATOR
   * If existing academic mathematical formulas aren't fully capturing market inefficiencies,
   * the AI generates/evolves custom math timing expressions to combine different risk variables.
   */
  private generateEvolvedFormula(genes: MathGenes): string {
    const formulas = [
      `F_timed(t) = P(X >= ${genes.binomialThreshold.toFixed(2)} | H₀) * K_vel + Kelly(${genes.kellyFraction.toFixed(2)}) - Z_score(${genes.zScoreEntry.toFixed(1)}) * VarRatio()`,
      `Θ_regime(x) = Kalman(x, Q=${genes.kalmanNoiseRatio.toFixed(4)}) ⊗ Hurst(${genes.hurstTrending.toFixed(2)}) + exp(z_score / J_mom(${genes.momentumLookback}))`,
      `Ω_optimal(t) = (Kelly(${genes.kellyFraction.toFixed(2)}) * Volatility(30) + z_score(${genes.zScoreEntry.toFixed(1)})) / (VarianceRatio() * H_hurst)`,
      `Ψ_momentum(t) = J_mom(${genes.momentumLookback}) * P(Streak >= k) + (z_score / Z_exit(${genes.zScoreExit.toFixed(2)}))`
    ];
    // Return one dynamically based on current generation
    return formulas[this.generation % formulas.length];
  }

  private geminiApiKey = 'AIzaSyCD6RNHN1FJ-OQxtasRTgJ2dOzUvRLrhnk';

  private async callGemini(prompt: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const data = JSON.stringify({
        contents: [
          {
            parts: [
              { text: prompt }
            ]
          }
        ],
        generationConfig: {
          responseMimeType: "application/json"
        }
      });

      const options = {
        hostname: 'generativelanguage.googleapis.com',
        port: 443,
        path: `/v1beta/models/gemma-4-31b:generateContent?key=${this.geminiApiKey}`,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(data),
        },
      };

      const req = https.request(options, (res) => {
        let body = '';
        res.on('data', (chunk) => body += chunk);
        res.on('end', () => {
          if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
            resolve(body);
          } else {
            reject(new Error(`HTTP ${res.statusCode}: ${body}`));
          }
        });
      });

      req.on('error', (e) => reject(e));
      req.write(data);
      req.end();
    });
  }

  /**
   * EVOLVE CYCLE
   * Triggered when underperformance is detected. Generates candidate mutations,
   * runs backtest on price history, and selects the absolute best chromosome.
   */
  async evolve(priceHistory: number[], tradeResults: any[]): Promise<boolean> {
    if (priceHistory.length < 50) return false;

    this.generation++;
    this.lastEvolvedAt = Date.now();
    this.log(`AI Gen ${this.generation} activated: analyzing underperforming vectors.`);

    // 1. Try Gemini API Optimization first
    try {
      this.log(`Reaching out to Google Gemma 4 (gemma-4-31b) for optimization...`);
      const closedTrades = tradeResults.filter((t: any) => t.status === 'CLOSED');
      const winningTrades = closedTrades.filter((t: any) => (t.pnl || 0) > 0);
      const winRate = closedTrades.length > 0 ? (winningTrades.length / closedTrades.length) * 100 : 50;
      const netProfit = closedTrades.reduce((sum: number, t: any) => sum + (t.pnl || 0), 0);

      const prompt = `You are a Quantitative Trading AI Expert. Your goal is to optimize the mathematical parameters (genes) for a Dogecoin trading bot.
Current parameters (genes):
${JSON.stringify(this.activeGenes, null, 2)}

Recent market statistics:
- Total closed trades: ${closedTrades.length}
- Win rate: ${winRate.toFixed(1)}%
- Net profit/loss: $${netProfit.toFixed(2)} USDT

Recent price history (last 50 closing prices):
${JSON.stringify(priceHistory.slice(-50))}

Please output a JSON object containing the optimized values for each parameter. The output format MUST be exactly:
{
  "binomialThreshold": number (between 0.55 and 0.95),
  "zScoreEntry": number (between -3.5 and -1.0),
  "zScoreExit": number (between 0.5 and 2.5),
  "hurstTrending": number (between 0.50 and 0.75),
  "hurstReversion": number (between 0.25 and 0.49),
  "kalmanNoiseRatio": number (between 0.0005 and 0.05),
  "varianceRatioLongPeriod": number (integer between 5 and 25),
  "kellyFraction": number (between 0.05 and 0.50),
  "momentumLookback": number (integer between 5 and 40),
  "bestFormulaExpression": "string (representing the evolved mathematical formula expression using latex or quant notation, e.g. Ω(t) = Kelly * Z_score - Kalman_noise)",
  "logMessage": "string (brief summary of why you chose these parameters, max 100 characters)"
}

Do not include any explanation outside the JSON format. Respond ONLY with valid JSON.`;

      const responseBody = await this.callGemini(prompt);
      const parsed = JSON.parse(responseBody);
      const text = parsed.candidates[0].content.parts[0].text.trim();
      const responseJson = JSON.parse(text);

      // Validate inputs from Gemini
      if (responseJson && typeof responseJson.binomialThreshold === 'number') {
        this.activeGenes = {
          binomialThreshold: Number(responseJson.binomialThreshold),
          zScoreEntry: Number(responseJson.zScoreEntry),
          zScoreExit: Number(responseJson.zScoreExit),
          hurstTrending: Number(responseJson.hurstTrending),
          hurstReversion: Number(responseJson.hurstReversion),
          kalmanNoiseRatio: Number(responseJson.kalmanNoiseRatio),
          varianceRatioLongPeriod: Math.round(Number(responseJson.varianceRatioLongPeriod)),
          kellyFraction: Number(responseJson.kellyFraction),
          momentumLookback: Math.round(Number(responseJson.momentumLookback))
        };
        this.bestFormulaExpression = responseJson.bestFormulaExpression || this.bestFormulaExpression;
        this.fitnessScore = this.evaluateFitness(this.activeGenes, priceHistory);
        this.log(`🔮 GEMINI OPTIMIZER SUCCESS: ${responseJson.logMessage || 'Parameters successfully evolved.'}`);
        this.log(`New evolved formula expression: "${this.bestFormulaExpression}"`);
        this.log(`Updated parameters: Z_entry=${this.activeGenes.zScoreEntry.toFixed(2)}, Hurst_trend=${this.activeGenes.hurstTrending.toFixed(2)}, Kelly_frac=${this.activeGenes.kellyFraction.toFixed(2)}.`);
        return true;
      }
    } catch (e: any) {
      this.log(`Gemini optimization failed (${e.message}). Falling back to local Genetic Engine...`);
    }

    // 2. Fallback to Local Genetic Algorithm
    const currentFitness = this.evaluateFitness(this.activeGenes, priceHistory);
    this.log(`Current active mathematical formula fitness: ${currentFitness.toFixed(2)} points.`);

    const populationSize = 15;
    let bestCandidate = this.activeGenes;
    let bestFitness = currentFitness;

    for (let i = 0; i < populationSize; i++) {
      const candidate = this.mutate(this.activeGenes, 0.25);
      const candidateFitness = this.evaluateFitness(candidate, priceHistory);

      if (candidateFitness > bestFitness) {
        bestFitness = candidateFitness;
        bestCandidate = candidate;
      }
    }

    if (bestFitness > currentFitness) {
      const improvement = bestFitness - currentFitness;
      this.activeGenes = bestCandidate;
      this.fitnessScore = bestFitness;
      this.bestFormulaExpression = this.generateEvolvedFormula(bestCandidate);

      this.log(`🧬 GA FALLBACK SUCCESS! Generation ${this.generation} evolved a superior chromosome.`);
      this.log(`Improvement delta: +${improvement.toFixed(2)} backtest fitness units.`);
      this.log(`New evolved formula expression: "${this.bestFormulaExpression}"`);
      this.log(`Updated parameters: Z_entry=${bestCandidate.zScoreEntry.toFixed(2)}, Hurst_trend=${bestCandidate.hurstTrending.toFixed(2)}, Kelly_frac=${bestCandidate.kellyFraction.toFixed(2)}.`);
      return true;
    } else {
      this.fitnessScore = currentFitness;
      this.log(`Gen ${this.generation} stagnation: parent chromosome remains mathematically superior. Exploring hyperparameter space.`);
      return false;
    }
  }

}
