export interface OrderBookSignal {
    obi: number;
    microPressure: number;
    wallSide: 'BUY' | 'SELL' | 'NONE';
    wallPrice: number;
    spread: number;
}

export class OrderBookSensor {
    private baseUrl = 'https://api.binance.com/api/v3/depth';

    async fetchOrderBookSnapshot(symbol: string, limit = 20): Promise<OrderBookSignal> {
        try {
            const response = await fetch(`${this.baseUrl}?symbol=${symbol.toUpperCase()}&limit=${limit}`);
            if (!response.ok) throw new Error(`OrderBook Fetch Failed: ${response.status}`);

            const data = await response.json() as { bids: [string, string][], asks: [string, string][] };
            return this.computeImbalance(data);
        } catch (error) {
            console.error('OrderBookSensor Error:', error);
            return { obi: 0, microPressure: 0, wallSide: 'NONE', wallPrice: 0, spread: 0 };
        }
    }

    private computeImbalance(data: { bids: [string, string][], asks: [string, string][] }): OrderBookSignal {
        const bids = data.bids.map(b => ({ price: parseFloat(b[0]), vol: parseFloat(b[1]) }));
        const asks = data.asks.map(a => ({ price: parseFloat(a[0]), vol: parseFloat(a[1]) }));

        if (bids.length === 0 || asks.length === 0) {
            return { obi: 0, microPressure: 0, wallSide: 'NONE', wallPrice: 0, spread: 0 };
        }

        const midPrice = (bids[0].price + asks[0].price) / 2;
        const spread = ((asks[0].price - bids[0].price) / bids[0].price) * 100;

        // 1. OBI Calculation
        const totalBidVol = bids.reduce((s, b) => s + b.vol, 0);
        const totalAskVol = asks.reduce((s, a) => s + a.vol, 0);
        const obi = (totalBidVol - totalAskVol) / (totalBidVol + totalAskVol);

        // 2. Micro Pressure Index (Weighted Imbalance)
        // Favoring levels 1-5 (3x weight) vs mid levels (2x) vs outer levels (1x)
        let weightedBid = 0;
        let weightedAsk = 0;

        for (let i = 0; i < bids.length; i++) {
            const weight = i < 5 ? 3 : i < 12 ? 2 : 1;
            if (bids[i]) weightedBid += bids[i].vol * weight;
            if (asks[i]) weightedAsk += asks[i].vol * weight;
        }
        const microPressure = (weightedBid - weightedAsk) / (weightedBid + weightedAsk);

        // 3. Wall Detection (> 3x average level size)
        const avgBidVol = totalBidVol / bids.length;
        const avgAskVol = totalAskVol / asks.length;

        let wallSide: 'BUY' | 'SELL' | 'NONE' = 'NONE';
        let wallPrice = 0;
        let maxWallSize = 0;

        // Check Bid Walls (Support)
        bids.forEach(b => {
            if (b.vol > avgBidVol * 3 && b.vol > maxWallSize) {
                wallSide = 'BUY';
                wallPrice = b.price;
                maxWallSize = b.vol;
            }
        });

        // Check Ask Walls (Resistance) - Overwrites if bigger
        asks.forEach(a => {
            if (a.vol > avgAskVol * 3 && a.vol > maxWallSize) {
                wallSide = 'SELL';
                wallPrice = a.price;
                maxWallSize = a.vol;
            }
        });

        return {
            obi: parseFloat(obi.toFixed(3)),
            microPressure: parseFloat(microPressure.toFixed(3)),
            wallSide,
            wallPrice,
            spread: parseFloat(spread.toFixed(4))
        };
    }
}