#ifndef AI_H
#define AI_H

#define INPUT_SIZE 784 //  28x28
#define HIDDEN_SIZE 128 //  Hidden layer nodes
#define OUTPUT_SIZE 10 //  Digits 0-9

typedef struct {
    float w1[INPUT_SIZE * HIDDEN_SIZE];  // Weights: Input -> Hidden
    float b1[HIDDEN_SIZE];               // Biases: Hidden
    float w2[HIDDEN_SIZE * OUTPUT_SIZE]; // Weights: Hidden -> Output
    float b2[OUTPUT_SIZE];               // Biases: Output
} NeuralNetwork;

typedef struct {
    float input[INPUT_SIZE];
    float hidden[HIDDEN_SIZE];
    float output[OUTPUT_SIZE];
} NetworkState;

void initDummyWeights(NeuralNetwork *net);
void runInference(const NeuralNetwork *net, NetworkState *state, const float *inputPixels);

#endif //  AI_H