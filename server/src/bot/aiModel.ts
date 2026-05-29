export interface NeuralNetworkWeights {
  inputToHidden: number[][]; // [hiddenSize][inputSize]
  hiddenToOutput: number[][]; // [outputSize][hiddenSize]
  hiddenBiases: number[];
  outputBiases: number[];
}

export class CustomNeuralNetwork {
  private inputSize: number;
  private hiddenSize: number;
  private outputSize: number;
  private learningRate: number;

  // Weights & Biases
  private weightsIH: number[][];
  private weightsHO: number[][];
  private biasH: number[];
  private biasO: number[];

  // Cache for backpropagation
  private lastInputs: number[] = [];
  private lastHidden: number[] = [];
  private lastOutputs: number[] = [];

  constructor(
    inputSize = 5,
    hiddenSize = 8,
    outputSize = 2,
    learningRate = 0.1
  ) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.learningRate = learningRate;

    // Initialize weights and biases randomly (-1 to 1)
    this.weightsIH = Array.from({ length: this.hiddenSize }, () =>
      Array.from({ length: this.inputSize }, () => Math.random() * 2 - 1)
    );
    this.weightsHO = Array.from({ length: this.outputSize }, () =>
      Array.from({ length: this.hiddenSize }, () => Math.random() * 2 - 1)
    );
    this.biasH = Array.from({ length: this.hiddenSize }, () => Math.random() * 2 - 1);
    this.biasO = Array.from({ length: this.outputSize }, () => Math.random() * 2 - 1);
  }

  // Sigmoid activation function
  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  // Derivative of sigmoid
  private sigmoidDerivative(y: number): number {
    return y * (1 - y);
  }

  // Forward propagation
  forward(inputs: number[]): number[] {
    this.lastInputs = [...inputs];

    // Compute Hidden Layer
    this.lastHidden = [];
    for (let h = 0; h < this.hiddenSize; h++) {
      let sum = this.biasH[h];
      for (let i = 0; i < this.inputSize; i++) {
        sum += inputs[i] * this.weightsIH[h][i];
      }
      this.lastHidden.push(this.sigmoid(sum));
    }

    // Compute Output Layer
    this.lastOutputs = [];
    for (let o = 0; o < this.outputSize; o++) {
      let sum = this.biasO[o];
      for (let h = 0; h < this.hiddenSize; h++) {
        sum += this.lastHidden[h] * this.weightsHO[o][h];
      }
      this.lastOutputs.push(this.sigmoid(sum));
    }

    return [...this.lastOutputs];
  }

  // Backpropagation to train with targets
  train(targets: number[]): void {
    if (this.lastInputs.length === 0) return;

    // 1. Calculate Output Layer Errors & Gradients
    const outputErrors: number[] = [];
    const outputGradients: number[] = [];
    for (let o = 0; o < this.outputSize; o++) {
      const error = targets[o] - this.lastOutputs[o];
      outputErrors.push(error);
      outputGradients.push(error * this.sigmoidDerivative(this.lastOutputs[o]));
    }

    // 2. Calculate Hidden Layer Errors & Gradients
    const hiddenErrors: number[] = [];
    const hiddenGradients: number[] = [];
    for (let h = 0; h < this.hiddenSize; h++) {
      let error = 0;
      for (let o = 0; o < this.outputSize; o++) {
        error += outputGradients[o] * this.weightsHO[o][h];
      }
      hiddenErrors.push(error);
      hiddenGradients.push(error * this.sigmoidDerivative(this.lastHidden[h]));
    }

    // 3. Update weights HO & biases O
    for (let o = 0; o < this.outputSize; o++) {
      this.biasO[o] += outputGradients[o] * this.learningRate;
      for (let h = 0; h < this.hiddenSize; h++) {
        this.weightsHO[o][h] += outputGradients[o] * this.lastHidden[h] * this.learningRate;
      }
    }

    // 4. Update weights IH & biases H
    for (let h = 0; h < this.hiddenSize; h++) {
      this.biasH[h] += hiddenGradients[h] * this.learningRate;
      for (let i = 0; i < this.inputSize; i++) {
        this.weightsIH[h][i] += hiddenGradients[h] * this.lastInputs[i] * this.learningRate;
      }
    }
  }

  // Reinforcement reward: positive reward trains the NN to prefer the winning action, negative trains to avoid it
  reinforce(inputs: number[], actionTaken: 'BUY' | 'SELL', profitPercent: number): void {
    this.forward(inputs); // Ensure cache is loaded

    // Create target output
    // Standard targets: index 0 = BUY target, index 1 = SELL target
    const currentOutputs = [...this.lastOutputs];
    const target = [...currentOutputs];

    const actionIdx = actionTaken === 'BUY' ? 0 : 1;
    const alternateIdx = actionTaken === 'BUY' ? 1 : 0;

    if (profitPercent > 0) {
      // Reward the action taken
      target[actionIdx] = 0.95; // Strong confidence target
      target[alternateIdx] = 0.05; // Drop alternate
    } else {
      // Penalize the action taken (train to NOT buy or NOT sell in this condition)
      target[actionIdx] = 0.05;
      target[alternateIdx] = 0.50; // Neutralize or prefer alternate slightly
    }

    this.train(target);
  }

  // Get current state of weights/biases for visualizer
  getWeightsAndNeurons(): NeuralNetworkWeights & {
    inputs: number[];
    hidden: number[];
    outputs: number[];
  } {
    return {
      inputToHidden: this.weightsIH,
      hiddenToOutput: this.weightsHO,
      hiddenBiases: this.biasH,
      outputBiases: this.biasO,
      inputs: this.lastInputs,
      hidden: this.lastHidden,
      outputs: this.lastOutputs,
    };
  }

  // Helper to normalize input values into -1 to 1 or 0 to 1 range
  static normalizeIndicators(
    rsi: number,
    macdHist: number,
    emaRatio: number, // short_ema / long_ema
    bbPosition: number, // where current price is relative to BB: (price - lower) / (upper - lower)
    botActivity: number // Normalized index of detected bot mass movements
  ): number[] {
    return [
      (rsi - 50) / 50, // Normalize 0..100 to -1..1
      Math.max(-1, Math.min(1, macdHist * 100)), // MACD Hist scaled
      (emaRatio - 1.0) * 100, // Percentage difference
      bbPosition * 2 - 1, // Scale 0..1 to -1..1
      Math.min(1, botActivity / 5) * 2 - 1 // Scale 0..5+ to -1..1
    ];
  }
}
