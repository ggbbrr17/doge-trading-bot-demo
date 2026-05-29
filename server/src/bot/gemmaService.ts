import { GoogleGenAI } from '@google/genai';
import * as dotenv from 'dotenv';

dotenv.config();

export interface GemmaSignal {
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  reason: string;
}

export class GemmaService {
  private ai: GoogleGenAI | null = null;
  private modelName = 'gemma-4-26b-a4b-it';

  constructor() {
    const apiKey = process.env.GEMINI_API_KEY;
    if (apiKey) {
      this.ai = new GoogleGenAI({ apiKey });
      console.log(`[GemmaService] Initialized with API key using model: ${this.modelName}`);
    } else {
      console.warn('[GemmaService] Warning: GEMINI_API_KEY not found in environment variables. Unified AI strategy will fall back to HOLD.');
    }
  }

  updateApiKey(apiKey: string) {
    if (apiKey) {
      this.ai = new GoogleGenAI({ apiKey });
      console.log(`[GemmaService] API key updated dynamically.`);
    }
  }

  async getSignal(
    symbol: string,
    currentPrice: number,
    indicators: { rsi: number; macdHist: number; emaRatio: number; bbPosition: number },
    candles: { time: number; open: number; high: number; low: number; close: number; volume: number }[]
  ): Promise<GemmaSignal> {
    if (!this.ai) {
      return {
        action: 'HOLD',
        confidence: 0.5,
        stopLossPercent: 2.0,
        takeProfitPercent: 1.5,
        reason: 'Gemma Service not initialized (missing GEMINI_API_KEY).',
      };
    }

    try {
      // 1. Programmatic SMC Detection
      const swings: { type: 'HIGH' | 'LOW'; price: number; index: number }[] = [];
      const fvgs: { type: 'BULLISH' | 'BEARISH'; bottom: number; top: number; index: number }[] = [];
      const orderBlocks: { type: 'BULLISH' | 'BEARISH'; open: number; high: number; low: number; close: number; index: number }[] = [];

      if (candles.length >= 3) {
        for (let i = 1; i < candles.length - 1; i++) {
          // Swing High (Local Peak)
          if (candles[i].high > candles[i - 1].high && candles[i].high > candles[i + 1].high) {
            swings.push({ type: 'HIGH', price: candles[i].high, index: i + 1 });
          }
          // Swing Low (Local Trough)
          if (candles[i].low < candles[i - 1].low && candles[i].low < candles[i + 1].low) {
            swings.push({ type: 'LOW', price: candles[i].low, index: i + 1 });
          }
        }

        for (let i = 2; i < candles.length; i++) {
          // Bullish FVG: high[i-2] < low[i]
          if (candles[i - 2].high < candles[i].low) {
            fvgs.push({ type: 'BULLISH', bottom: candles[i - 2].high, top: candles[i].low, index: i + 1 });
          }
          // Bearish FVG: low[i-2] > high[i]
          if (candles[i - 2].low > candles[i].high) {
            fvgs.push({ type: 'BEARISH', bottom: candles[i].high, top: candles[i - 2].low, index: i + 1 });
          }
        }

        const avgBodySize = candles.reduce((sum, c) => sum + Math.abs(c.close - c.open), 0) / candles.length;
        for (let i = 1; i < candles.length; i++) {
          const bodySize = Math.abs(candles[i].close - candles[i].open);
          
          // Bullish OB: strong up close breaking previous high
          if (candles[i].close > candles[i].open && bodySize > avgBodySize && candles[i].close > candles[i - 1].high && candles[i - 1].close < candles[i - 1].open) {
            orderBlocks.push({
              type: 'BULLISH',
              open: candles[i - 1].open,
              high: candles[i - 1].high,
              low: candles[i - 1].low,
              close: candles[i - 1].close,
              index: i
            });
          }
          // Bearish OB: strong down close breaking previous low
          if (candles[i].close < candles[i].open && bodySize > avgBodySize && candles[i].close < candles[i - 1].low && candles[i - 1].close > candles[i - 1].open) {
            orderBlocks.push({
              type: 'BEARISH',
              open: candles[i - 1].open,
              high: candles[i - 1].high,
              low: candles[i - 1].low,
              close: candles[i - 1].close,
              index: i
            });
          }
        }
      }

      // Format SMC findings for LLM
      const formattedSwings = swings.map(s => `- Swing ${s.type} detected at $${s.price.toFixed(5)} (Candle ${s.index})`).join('\n') || '- No clear swing points found';
      const formattedFVGs = fvgs.map(f => `- ${f.type} FVG (Imbalance) gap from $${f.bottom.toFixed(5)} to $${f.top.toFixed(5)} (Mitigation range, Candle ${f.index})`).join('\n') || '- No active imbalances (FVG) found';
      const formattedOBs = orderBlocks.map(ob => `- ${ob.type} Order Block (Last opposite candle before expansion) at O=${ob.open.toFixed(5)}, H=${ob.high.toFixed(5)}, L=${ob.low.toFixed(5)}, C=${ob.close.toFixed(5)} (Mitigation zone, Candle ${ob.index})`).join('\n') || '- No active Order Blocks found';

      // Format candles for clear prompt interpretation
      const formattedCandles = candles.map((c, i) => {
        const bodySize = Math.abs(c.close - c.open);
        const upperWick = c.high - Math.max(c.open, c.close);
        const lowerWick = Math.min(c.open, c.close) - c.low;
        const wickRatio = bodySize > 0 ? (upperWick + lowerWick) / bodySize : 10;
        return `Candle ${i + 1} (t=${c.time}): O=${c.open.toFixed(5)}, H=${c.high.toFixed(5)}, L=${c.low.toFixed(5)}, C=${c.close.toFixed(5)}, V=${c.volume.toFixed(0)} | Body=${bodySize.toFixed(5)}, UpperWick=${upperWick.toFixed(5)}, LowerWick=${lowerWick.toFixed(5)}, Wick/Body Ratio=${wickRatio.toFixed(2)}`;
      }).join('\n');

      const prompt = `
You are an expert quantitative trading analyst embodying the trading methodology of "Esteban Pérez" from the YouTube channel "Bitcoin Hoy" combined with elite **Smart Money Concepts (SMC)** and **Order Flow** (Institutional Trading) strategies.
Your goal is to perform a detailed **Market Structure & Order Flow** analysis for the asset ${symbol} and output a trading decision: BUY, SELL, or HOLD.

### ADVANCED INSTITUTIONAL TRADING & ORDER FLOW RULES:
1. **Market Structure (BOS / CHoCH)**:
   - **CHoCH (Change of Character)**: The first break of the last structural swing point. Indicates a potential trend reversal. Look to enter in the opposite direction once confirmed.
   - **BOS (Break of Structure)**: Continuation of the current structural trend (breaking higher highs in uptrend, lower lows in downtrend).
2. **Order Blocks (OB)**:
   - **Bullish OB**: The last down-candle before a high-volume upward expansion that breaks structure. When price retraces/mitigates this zone, buy with a stop loss just below the block's low.
   - **Bearish OB**: The last up-candle before a high-volume downward expansion. When price retraces/mitigates this zone, sell with a stop loss just above the block's high.
3. **Imbalances & Fair Value Gaps (FVG)**:
   - Gaps in price delivery across a 3-candle sequence. These act as powerful liquidity magnets. Price almost always returns to fill/mitigate these gaps before continuing its trend.
4. **Rejection Wicks (Mechas de Rechazo)**: Look for long candle shadows sweeping key swing levels (liquidity pools) and closing back inside, showing immediate institutional rejection.
5. **Strict Risk Management**:
   - Stop Loss: Place immediately behind the invalidation level (just beyond the low of the Bullish OB or high of the Bearish OB).
   - Take Profit: Target the next major unmitigated Order Block or major swing level.

Please also use Google Search to find:
1. The latest real-time Bitcoin/Crypto market news, macroeconomic factors, and global market sentiment affecting price action today.
2. Any very recent specific technical levels, order blocks, or warnings shared by Esteban Pérez on "Bitcoin Hoy" or "Crecetrader" within the last 24 hours.

Here is the real-time technical status of our asset:
- Current Price: $${currentPrice}
- RSI: ${indicators.rsi.toFixed(2)} (Overbought > 70, Oversold < 30)
- MACD Histogram: ${indicators.macdHist.toFixed(5)}
- EMA Ratio (Short vs Long): ${indicators.emaRatio.toFixed(4)}
- Bollinger Band Position: ${indicators.bbPosition.toFixed(2)} (0 = Lower Band, 1 = Upper Band)

### PROGRAMMATICALLY DETECTED SMC STRUCTURAL CORE OUTPUT:
**Key Swing points:**
${formattedSwings}

**Imbalances & Fair Value Gaps (FVG):**
${formattedFVGs}

**Order Blocks (OB) Mitigation Zones:**
${formattedOBs}

Here is the raw data of the last 15 candles (from oldest to newest):
${formattedCandles}

### YOUR MISSION:
1. Analyze the **Market Structure**. Is the price showing CHoCH (Change of Character) or BOS (Break of Structure)? Is the trend structure bullish, bearish, or ranging?
2. Locate the active **Order Blocks** and **Fair Value Gaps (FVGs)** from the detected SMC data relative to the current price. Is the price currently mitigating a Bullish or Bearish OB? Is it filling an imbalance?
3. Review the candle wicks (mechas) of the recent candles for liquidity sweeps or rejection wicks.
4. Combine this structural SMC analysis with your Google Search news findings and Esteban Pérez's recent outlook.
5. Output a trading signal (BUY, SELL, or HOLD) aligning perfectly with this conformed institutional strategy and strict risk management.
6. Calculate the exact stopLossPercent (e.g. 1.5) and takeProfitPercent (e.g. 3.0) dynamically. The stop loss should be tightly behind the valid Order Block or swing low/high, and the take profit should target the next liquidity pool (swing point) or unmitigated FVG/OB.

Your response must be a single, raw JSON object (with no markdown wrappers, no backticks, and no extra text) matching this schema exactly:
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": number, // between 0.0 and 1.0
  "stopLossPercent": number, // positive float representing % (e.g. 1.25)
  "takeProfitPercent": number, // positive float representing % (e.g. 2.50)
  "reason": "Detailed step-by-step analysis in Spanish. Point out: 1) the market structure trend (CHoCH/BOS), 2) unmitigated Order Blocks (OB) and imbalances (FVG) identified near current price, 3) recent candle wick/liquidity sweeps, 4) news findings, and 5) how the risk invalidation level (stop loss) is placed relative to the mitigated Order Block according to SMC rules."
}
`;

      const response = await this.ai.models.generateContent({
        model: this.modelName,
        contents: prompt,
        config: {
          tools: [{ googleSearch: {} }],
        },
      });

      const responseText = response.text || '';
      console.log(`[GemmaService] Raw response received: ${responseText}`);

      // Strip markdown codeblock backticks if present
      let cleanText = responseText.trim();
      if (cleanText.startsWith('```')) {
        cleanText = cleanText.replace(/^```json\s*/i, '').replace(/```\s*$/g, '').trim();
      }

      const signal: GemmaSignal = JSON.parse(cleanText);
      
      // Validate response structure
      if (['BUY', 'SELL', 'HOLD'].includes(signal.action) && 
          typeof signal.confidence === 'number' && 
          typeof signal.stopLossPercent === 'number' && 
          typeof signal.takeProfitPercent === 'number' && 
          typeof signal.reason === 'string') {
        return signal;
      }

      throw new Error('Invalid JSON schema returned by Gemma 4');
    } catch (error: any) {
      console.error(`[GemmaService] Error fetching Gemma 4 signal: ${error.message}`);
      return {
        action: 'HOLD',
        confidence: 0.5,
        stopLossPercent: 2.0,
        takeProfitPercent: 1.5,
        reason: `Failed to generate Gemma signal: ${error.message}`,
      };
    }
  }
}
