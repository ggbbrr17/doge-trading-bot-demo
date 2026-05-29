import { spawn } from 'child_process';
import path from 'path';

export interface HMMResult {
    status: string;
    current_regime?: string;
    current_state_id?: number;
    state_descriptions?: Record<string, string>;
    observations_count?: number;
    message?: string;
}

export class HMMService {
    private pythonScriptPath: string;

    constructor() {
        this.pythonScriptPath = path.join(__dirname, '..', '..', 'python', 'hmm_regimes.py');
    }

    /**
     * Determines the current market regime using an HMM in Python.
     * Optionally accepts recent closing prices and volumes to feed into the model.
     * If empty arrays are passed, the Python script fetches data from yfinance.
     * @param closes Array of recent closing prices
     * @param volumes Array of recent volumes
     */
    public async getCurrentRegime(closes: number[] = [], volumes: number[] = []): Promise<HMMResult> {
        return new Promise((resolve, reject) => {
            const pythonProcess = spawn('python3', [this.pythonScriptPath]);

            let dataString = '';
            let errorString = '';

            pythonProcess.stdout.on('data', (data) => {
                dataString += data.toString();
            });

            pythonProcess.stderr.on('data', (data) => {
                errorString += data.toString();
            });

            pythonProcess.on('close', (code) => {
                if (code !== 0) {
                    console.error(`Python HMM script exited with code ${code}`);
                    console.error(errorString);
                    reject(new Error(`HMM Script Failed: ${errorString}`));
                    return;
                }

                try {
                    const result = JSON.parse(dataString);
                    resolve(result);
                } catch (e: any) {
                    console.error('Failed to parse HMM output', dataString);
                    reject(new Error(`Invalid JSON from HMM: ${e.message}`));
                }
            });

            // If we have data, send it via stdin. Otherwise, just close stdin to let Python fetch it.
            if (closes.length > 0 && volumes.length > 0) {
                const inputJson = JSON.stringify({ closes, volumes });
                pythonProcess.stdin.write(inputJson);
            }
            pythonProcess.stdin.end();
        });
    }
}

// Export a singleton instance for simplicity
export const hmmService = new HMMService();
