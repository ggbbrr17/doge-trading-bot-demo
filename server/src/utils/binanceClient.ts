import * as crypto from 'crypto';

export interface BinanceConfig {
  apiKey: string;
  apiSecret: string;
  isTestnet: boolean;
  marketType: 'SPOT' | 'FUTURES';
}

export class BinanceClient {
  private apiKey: string;
  private apiSecret: string;
  private baseUrl: string;
  private isFutures: boolean;

  constructor(config: BinanceConfig) {
    this.apiKey = config.apiKey;
    this.apiSecret = config.apiSecret;
    this.isFutures = config.marketType === 'FUTURES';
    
    if (this.isFutures) {
      this.baseUrl = config.isTestnet
        ? 'https://demo-fapi.binance.com'
        : 'https://fapi.binance.com';
    } else {
      this.baseUrl = config.isTestnet
        ? 'https://testnet.binance.vision'
        : 'https://api.binance.com';
    }
  }

  // Generate HMAC SHA256 signature
  private sign(queryString: string): string {
    return crypto
      .createHmac('sha256', this.apiSecret)
      .update(queryString)
      .digest('hex');
  }

  // Get server time from Binance to prevent sync issues
  async getServerTime(): Promise<number> {
    const endpoint = this.isFutures ? '/fapi/v1/time' : '/api/v3/time';
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = (await response.json()) as { serverTime: number };
      return data.serverTime;
    } catch (error) {
      console.error('Error fetching Binance server time:', error);
      return Date.now();
    }
  }

  // Make signed private request
  private async signedRequest(
    path: string,
    method: 'GET' | 'POST' | 'DELETE' = 'GET',
    params: Record<string, string> = {}
  ): Promise<any> {
    if (!this.apiKey || !this.apiSecret) {
      throw new Error('API Key and Secret Key are required for signed operations.');
    }

    const serverTime = await this.getServerTime();
    const queryParams = new URLSearchParams({
      ...params,
      timestamp: serverTime.toString(),
      recvWindow: '10000', // larger window to account for network delays on Windows
    });

    const signature = this.sign(queryParams.toString());
    queryParams.append('signature', signature);

    const url = `${this.baseUrl}${path}?${queryParams.toString()}`;
    const headers = {
      'X-MBX-APIKEY': this.apiKey,
      'Content-Type': 'application/x-www-form-urlencoded',
    };

    try {
      const response = await fetch(url, { method, headers });
      const responseText = await response.text();
      
      let data;
      try {
        data = JSON.parse(responseText);
      } catch (e) {
        throw new Error(`Failed to parse Binance response: ${responseText}`);
      }

      if (!response.ok) {
        throw new Error(
          `Binance API Error (${response.status}): ${data.msg || JSON.stringify(data)}`
        );
      }

      return data;
    } catch (error: any) {
      console.error(`Signed request error [${method} ${path}]:`, error.message);
      throw error;
    }
  }

  // Public endpoint: Get recent Klines (Candlesticks)
  async getKlines(symbol: string, interval: string, limit: number = 100): Promise<any[]> {
    const endpoint = this.isFutures ? '/fapi/v1/klines' : '/api/v3/klines';
    const url = `${this.baseUrl}${endpoint}?symbol=${symbol.toUpperCase()}&interval=${interval}&limit=${limit}`;
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return (await response.json()) as any[];
    } catch (error) {
      console.error(`Error fetching Klines for ${symbol}:`, error);
      throw error;
    }
  }

  // Public endpoint: Get ticker price
  async getTickerPrice(symbol: string): Promise<number> {
    const endpoint = this.isFutures ? '/fapi/v1/ticker/price' : '/api/v3/ticker/price';
    const url = `${this.baseUrl}${endpoint}?symbol=${symbol.toUpperCase()}`;
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = (await response.json()) as { price: string };
      return parseFloat(data.price);
    } catch (error) {
      console.error(`Error fetching price for ${symbol}:`, error);
      throw error;
    }
  }

  // Private endpoint: Get account balances
  async getAccountInfo(): Promise<any> {
    const endpoint = this.isFutures ? '/fapi/v2/account' : '/api/v3/account';
    return this.signedRequest(endpoint, 'GET');
  }

  // Private endpoint: Place order
  async placeOrder(
    symbol: string,
    side: 'BUY' | 'SELL',
    type: 'LIMIT' | 'MARKET',
    quantity: number,
    price?: number
  ): Promise<any> {
    const endpoint = this.isFutures ? '/fapi/v1/order' : '/api/v3/order';
    
    // Futures DOGE contracts require integer quantities (0 decimals)
    const formattedQty = this.isFutures 
      ? Math.round(quantity).toString() 
      : quantity.toFixed(2);

    const params: Record<string, string> = {
      symbol: symbol.toUpperCase(),
      side: side.toUpperCase(),
      type: type.toUpperCase(),
      quantity: formattedQty,
    };

    if (type === 'LIMIT') {
      if (price === undefined) {
        throw new Error('Price is required for LIMIT orders.');
      }
      params.price = price.toFixed(5); // DOGE USDT supports 5 decimal places for price
      params.timeInForce = 'GTC'; // Good Til Canceled
    }

    return this.signedRequest(endpoint, 'POST', params);
  }

  // Private endpoint: Get open orders
  async getOpenOrders(symbol: string): Promise<any[]> {
    const endpoint = this.isFutures ? '/fapi/v1/openOrders' : '/api/v3/openOrders';
    return this.signedRequest(endpoint, 'GET', { symbol: symbol.toUpperCase() });
  }

  // Private endpoint: Cancel order
  async cancelOrder(symbol: string, orderId: number): Promise<any> {
    const endpoint = this.isFutures ? '/fapi/v1/order' : '/api/v3/order';
    return this.signedRequest(endpoint, 'DELETE', {
      symbol: symbol.toUpperCase(),
      orderId: orderId.toString(),
    });
  }

  // Private endpoint: Set leverage (Futures only)
  async setLeverage(symbol: string, leverage: number): Promise<any> {
    if (!this.isFutures) return;
    return this.signedRequest('/fapi/v1/leverage', 'POST', {
      symbol: symbol.toUpperCase(),
      leverage: leverage.toString(),
    });
  }
}
