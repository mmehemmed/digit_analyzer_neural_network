#include "raylib.h"
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "ai.h"

#define CANVAS_SIZE 448
#define GRID_SIZE 28
#define BLOCK_SIZE (CANVAS_SIZE / GRID_SIZE) // 16 pixels per cell

RenderTexture2D screen;
NeuralNetwork net;
NetworkState aiState;

void InitCanvas(void);
void UpdateDrawing(void);
float* GetNormalizedPixelData(void);
void DrawNetworkVisualizer(const NetworkState *state, const NeuralNetwork *net);

int main(void) {
    InitWindow(CANVAS_SIZE * 4, CANVAS_SIZE * 2, "AI Digit Recognizer");
    SetTargetFPS(120);

    initDummyWeights(&net);

    InitCanvas();

    while (!WindowShouldClose()) {
        // Handle input & update drawing state
        UpdateDrawing();

        // Render panels
        BeginDrawing();
            ClearBackground(BLACK);

            // Main Canvas
            DrawTextureRec(
                screen.texture, 
                (Rectangle){ 0, 0, (float)CANVAS_SIZE, (float)-CANVAS_SIZE }, 
                (Vector2){ 0, 0 }, 
                WHITE
            );

            DrawNetworkVisualizer(&aiState, &net);
        EndDrawing();


        float *pixels = GetNormalizedPixelData();
        runInference(&net, &aiState, pixels);
    }



    UnloadRenderTexture(screen);
    CloseWindow();
    return 0;
}

void InitCanvas(void) {
    screen = LoadRenderTexture(CANVAS_SIZE, CANVAS_SIZE);
    BeginTextureMode(screen);
        ClearBackground(BLACK);
    EndTextureMode();
}

void UpdateDrawing(void) {
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 mouse = GetMousePosition();

        // Restrict drawing input to Canvas bounds
        if (mouse.x >= 0 && mouse.x < CANVAS_SIZE && mouse.y >= 0 && mouse.y < CANVAS_SIZE) {
            BeginTextureMode(screen);
                Vector2 drawPos = { mouse.x, mouse.y };
                DrawCircleV(drawPos, 12, WHITE);
            EndTextureMode();
        }
    }

    // Press 'C' to clear drawing canvas
    if (IsKeyPressed(KEY_C)) {
        BeginTextureMode(screen);
            ClearBackground(BLACK);
        EndTextureMode();
    }
}

float* GetNormalizedPixelData(void) {
    Image image = LoadImageFromTexture(screen.texture);
    Color *srcPixels = LoadImageColors(image);

    static float normalizedData[GRID_SIZE * GRID_SIZE];

    for (size_t gridY = 0; gridY < GRID_SIZE; gridY++) {
        for (size_t gridX = 0; gridX < GRID_SIZE; gridX++) {
            float totalIntensity = 0.0f;

            // Average brightness across the 16x16 block
            for (size_t srcY = 0; srcY < BLOCK_SIZE; srcY++) {
                for (size_t srcX = 0; srcX < BLOCK_SIZE; srcX++) {
                    int x = (int)(gridX * BLOCK_SIZE + srcX);
                    int y = (int)(gridY * BLOCK_SIZE + srcY);

                    totalIntensity += (float)srcPixels[y * CANVAS_SIZE + x].r;
                }
            }

            // Normalize pixel values to 0.0 - 1.0 range
            float average = totalIntensity / (float)(BLOCK_SIZE * BLOCK_SIZE);
            normalizedData[gridY * GRID_SIZE + gridX] = average / 255.0f;
        }
    }

    UnloadImageColors(srcPixels);
    UnloadImage(image);

    return normalizedData;
}

void DrawNetworkVisualizer(const NetworkState *state, const NeuralNetwork *net){
    // Panel Offsets
    float p2x = CANVAS_SIZE;
    float p3y = CANVAS_SIZE;

    float inputColX = p2x + 90.0f;
    float hiddenColX = p2x + CANVAS_SIZE;
    float outputColX = p2x + CANVAS_SIZE + CANVAS_SIZE - 100.0f;


    //  DRAW CONNECTION LINES
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        if (state->hidden[i] <= 0.05f) continue;

        float hy = 20.0f + i * ((CANVAS_SIZE * 2 - 40.0f) / (float)HIDDEN_SIZE);

        for (int k = 0; k < OUTPUT_SIZE; k++)
        {
            float weight = net->w2[i * OUTPUT_SIZE + k];
            float signal = state->hidden[i] * weight;
            if ( signal > 0.05f || signal < -0.05f) {
                float oy = 40.0f + k * ((CANVAS_SIZE * 2 - 80.0f)) / (float)OUTPUT_SIZE;
                Color lineCol = (signal > 0.0f) ? SKYBLUE : RED;
                float alpha = fminf(fabsf(signal) * 2.0f, 1.0f);

                DrawLineEx((Vector2){hiddenColX,hy}, (Vector2){outputColX,oy},5.0f, ColorAlpha(lineCol,alpha));
            }
        }
    }
    
    // DRAW INPUT LAYER

    for (int y = 0; y < 14; y++)
    {
        for (int x = 0; x < 14; x++)
        {
            float val = state->input[((13-y) * 2) * 28 + (x * 2)];
            float nx = inputColX - 35.0f + (x * 5.0f);
            float ny = 100.0f + (y * 5.0f);

            DrawRectangle((int)nx,(int)ny,4,4,ColorAlpha(WHITE,val));
        }
        
    }
    
    // DRAW HIDDEN LAYER NODES

    for (int j = 0; j < HIDDEN_SIZE; j++)
    {
        float val = state->hidden[j];
        float hy = 20.0f + j * ((CANVAS_SIZE * 2 - 40.0f) / (float)HIDDEN_SIZE);

        Color nodeCol = ColorAlpha(GREEN, fmaxf(0.15f, val));
        DrawCircleV((Vector2){hiddenColX,hy}, 3.0f, nodeCol);
    }
    
    // Draw OUTPUT LAYER NODES

    int bestClass = 0;
    float maxProb = 0.0f;

    for (int k = 0; k < OUTPUT_SIZE; k++)
    {
        float prob = state->output[k];
        float oy = 40.0f + k * ((CANVAS_SIZE * 2 - 80.0f) / (float)OUTPUT_SIZE);

        if (prob > maxProb)
        {
            maxProb = prob;
            bestClass = k;
        }

        DrawCircleV((Vector2){outputColX,oy},18.0f,ColorAlpha(GREEN,fmaxf(0.2f,prob)));
        DrawText(TextFormat("%d", k),outputColX-4,oy-9,20,BLACK);

        float barX = 60.0f;
        float barY = oy/2 + CANVAS_SIZE;
        float maxBarWidth = 220.0f;

        DrawText(TextFormat("%d", k), 30, (int)barY - 2, 20, WHITE);
        DrawRectangle((int)barX, (int)barY, (int)maxBarWidth, 16, DARKGRAY);
        DrawRectangle((int)barX, (int)barY, (int)(maxBarWidth * prob), 16, GREEN);
        DrawText(TextFormat("%.1f%%", prob * 100.0f), (int)(barX + maxBarWidth + 10.0f), (int)barY - 2, 16, LIGHTGRAY);
    }

    DrawText(TextFormat("GUESS: %d", bestClass), (int)p2x, CANVAS_SIZE*2-120.0f, 32, YELLOW);
    DrawText(TextFormat("Confidence: %.1f%%", maxProb * 100.0f), (int)p2x, CANVAS_SIZE*2-50.0f, 18, WHITE);
}