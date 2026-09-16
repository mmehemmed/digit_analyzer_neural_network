#include "ai.h"
#include <math.h>
#include <stdlib.h>

void initDummyWeights(NeuralNetwork *net){
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++)
    {
        net->w1[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) net->b1[i] = 0.0f;
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++)
    {
        net->w2[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) net->b2[i] = 0.0f;
}

void runInference(const NeuralNetwork *net, NetworkState *state, const float *inputPixels){
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        state->input[i] = inputPixels[i];
    }

    for (int j = 0; j < HIDDEN_SIZE; j++)
    {
        float sum = net->b1[j];  
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            sum += state->input[i] * net->w1[i * HIDDEN_SIZE + j];   // Computes Dot Product
        }
        state->hidden[j] = sum > 0.0f ? sum : 0.0f;  //ReLU
        
    }
    
    float maxLogit = -1e9f;
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        float sum = net->b2[k];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += state->hidden[j] * net->w2[j * OUTPUT_SIZE + k];
        }
        state->output[k] = sum;
        if (sum > maxLogit) maxLogit = sum; // Track max for Softmax stability
    }
    
    float sumExp = 0.0f;
    for (int k = 0; k < OUTPUT_SIZE; k++)
    {
        state->output[k] = expf(state->output[k] - maxLogit);
        sumExp += state->output[k];
    }
    for (int k = 0; k < OUTPUT_SIZE; k++)
    {
        state->output[k] /= sumExp;
    }
}