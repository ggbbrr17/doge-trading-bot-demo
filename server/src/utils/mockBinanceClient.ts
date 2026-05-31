export interface MockConfig {
  apiKey?: string;
  apiSecret?: string;
  isTestnet?: boolean;
  marketType?: 'SPOT' | 'FUTURES';
}

let ORDER_ID_SEQ = 100000;

export class MockBinanceClient {
  private isFutures: boolean;
  private marketType: 'SPOT' | 'FUTURES';

  constructor(config: MockConfig = {}) {
    this.isFutures = (config.marketType === 'FUTURES');
    this.marketType = config.marketType || 'SPOT';
  }

  async getKlines(symbol: string, interval: string, limit: number = 100): Promise<any[]> {
    // return synthetic klines array compatible with Binance response
    const now = Date.now();
    const out: any[] = [];
    let price = 0.42;
    for (let i = 0; i < limit; i++) {
      const open = price;
      const change = (Math.random() - 0.5) * 0.002 * price;
      const close = Math.max(0.00001, open + change);
      const high = Math.max(open, close) + Math.random() * 0.0005 * price;
      const low = Math.min(open, close) - Math.random() * 0.0005 * price;
      const vol = Math.random() * 100000 + 20000;
      out.push([now - (limit - i) * 60000, open.toFixed(6), high.toFixed(6), low.toFixed(6), close.toFixed(6), vol.toString()]);
      price = close;
    }
    return out;
  }

  async getTickerPrice(symbol: string): Promise<number> {
    return 0.42 + (Math.random() - 0.5) * 0.002;
  }

  async getOrderBook(symbol: string, limit: number = 5): Promise<any> {
    const price = await this.getTickerPrice(symbol);
    const bids = [] as [string, string][];
    const asks = [] as [string, string][];
    for (let i = 0; i < limit; i++) {
      bids.push([(price - 0.0001 * i).toFixed(6), (Math.random() * 1000 + 100).toFixed(0)]);
      asks.push([(price + 0.0001 * i).toFixed(6), (Math.random() * 1000 + 100).toFixed(0)]);
    }
    return { bids, asks };
  }

  async getAccountInfo(): Promise<any> {
    // return fake balances and no positions
    return {
      assets: [{ asset: 'USDT', availableBalance: '10000.00' }],
      positions: []
    };
  }

  async placeOrder(
    symbol: string,
    side: 'BUY' | 'SELL',
    type: 'LIMIT' | 'MARKET',
    quantity: number,
    price?: number,
    reduceOnly?: boolean,
    positionSide?: 'LONG' | 'SHORT'
  ): Promise<any> {
    // Simulate immediate market fill
    const orderId = ++ORDER_ID_SEQ;
    const avgPrice = price || (0.42 + (Math.random() - 0.5) * 0.002);
    return {
      orderId,
      executedQty: quantity.toString(),
      origQty: quantity.toString(),
      avgPrice: avgPrice.toString(),
      fills: [{ price: avgPrice.toString(), qty: quantity.toString(), commission: '0' }]
    };
  }

  async getOpenOrders(symbol: string): Promise<any[]> {
    return [];
  }

  async cancelOrder(symbol: string, orderId: number): Promise<any> {
    return { success: true };
  }

  async setLeverage(symbol: string, leverage: number): Promise<any> {
    return { success: true };
  }
}
