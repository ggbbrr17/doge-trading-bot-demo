// persistence.ts
import fs from 'fs';
import path from 'path';

const TRADES_FILE = path.join(__dirname, '../trades.json');

export const saveTradeToHistory = (trade: any) => {
    try {
        let history = [];
        if (fs.existsSync(TRADES_FILE)) {
            const data = fs.readFileSync(TRADES_FILE, 'utf8');
            history = JSON.parse(data);
        }

        // Agregamos metadatos matemáticos extra para el análisis futuro
        const tradeWithMeta = {
            ...trade,
            recordedAt: new Date().toISOString(),
            marketCondition: "Analyzed"
        };

        history.push(tradeWithMeta);
        fs.writeFileSync(TRADES_FILE, JSON.stringify(history));
        console.log(`[PERSISTENCE] Trade ${trade.id} saved to trades.json`);
    } catch (error) {
        console.error('[PERSISTENCE ERROR] could not save trade:', error);
    }
};

// Llama a saveTradeToHistory cada vez que una operación se abra o cierre en tu motor principal.