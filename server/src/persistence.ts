// persistence.ts
import mongoose from 'mongoose';

const TradeSchema = new mongoose.Schema({
    id: { type: String, unique: true },
    symbol: String,
    side: String,
    type: String,
    price: Number,
    quantity: Number,
    amount: Number,
    timestamp: Number,
    status: String,
    pnl: Number,
    pnlPercent: Number,
    exitPrice: Number,
    exitTimestamp: Number,
    reason: String,
    targetSL: Number,
    targetTP: Number,
    recordedAt: { type: Date, default: Date.now },
    marketCondition: String
});

export const TradeModel = mongoose.model('Trade', TradeSchema);

const TradingLessonSchema = new mongoose.Schema({
    tradeId: String,
    side: String,
    pnl: Number,
    pnlPercent: Number,
    outcome: String, // 'WIN' | 'LOSS'
    reasonForClose: String,
    llmAnalysis: String,
    createdAt: { type: Date, default: Date.now }
});

export const TradingLessonModel = mongoose.model('TradingLesson', TradingLessonSchema);

export const saveTradeToHistory = async (trade: any) => {
    try {
        // Usamos upsert para actualizar si el ID ya existe o crear uno nuevo
        await TradeModel.findOneAndUpdate(
            { id: trade.id },
            { ...trade, recordedAt: new Date(), marketCondition: "Analyzed" },
            { upsert: true, new: true }
        );
        console.log(`[MONGODB] Trade ${trade.id} persisted to database.`);
    } catch (error) {
        console.error('[MONGODB ERROR] could not save trade:', error);
    }
};

// Llama a saveTradeToHistory cada vez que una operación se abra o cierre en tu motor principal.