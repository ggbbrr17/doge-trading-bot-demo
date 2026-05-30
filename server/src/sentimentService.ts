import * as https from 'https';

export interface SentimentSignal {
    score: number; // -1 to 1
    summary: string;
    isCriticalNegative: boolean;
}

export class SentimentService {
    private geminiApiKey = process.env.GEMINI_API_KEY || '';

    updateApiKey(apiKey: string) {
        this.geminiApiKey = apiKey;
    }

    async fetchSentiment(): Promise<SentimentSignal> {
        if (!this.geminiApiKey) {
            return { score: 0, summary: "Sentiment engine offline (API Key missing)", isCriticalNegative: false };
        }

        const prompt = `Analyze the current social and news sentiment for Dogecoin (DOGE) and Bitcoin (BTC) based on the latest available market data and reliable crypto news sources. 
    Identify extreme events (hacks, regulatory bans, systemic crashes). 
    Return ONLY a JSON object:
    {
      "score": number (from -1.0 extremely bearish to 1.0 extremely bullish),
      "summary": "string (brief summary of findings, max 150 chars)",
      "isCriticalNegative": boolean (true ONLY if there is a systemic crash, regulatory ban, or major exchange hack)
    }`;

        try {
            const responseBody = await this.callGemini(prompt);
            const parsed = JSON.parse(responseBody);
            const text = parsed.candidates[0].content.parts[0].text.trim();
            // Filter out markdown backticks if present
            const jsonStr = text.replace(/```json|```/g, '');
            return JSON.parse(jsonStr) as SentimentSignal;
        } catch (error) {
            console.error("SentimentService Error:", error);
            return { score: 0, summary: "Sentiment analysis unavailable", isCriticalNegative: false };
        }
    }

    private async callGemini(prompt: string): Promise<string> {
        return new Promise((resolve, reject) => {
            const data = JSON.stringify({
                contents: [{ parts: [{ text: prompt }] }],
                generationConfig: { responseMimeType: "application/json" }
            });

            const options = {
                hostname: 'generativelanguage.googleapis.com',
                port: 443,
                path: `/v1beta/models/gemma-4-26b-a4b-it:generateContent?key=${this.geminiApiKey}`,
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
}