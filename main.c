#include "raylib.h"
#include <stddef.h>

int windowSize = 28 * 16; // 448
RenderTexture2D screen;


float* getImagePixelData();
void draw();



int main(void) {

    InitWindow(windowSize*3, windowSize, "AI Number");

    SetTargetFPS(120);

    screen = LoadRenderTexture(windowSize,windowSize);
    
    BeginTextureMode(screen);
        ClearBackground(BLACK);
    EndTextureMode();

    float *imageData = NULL;

    while (!WindowShouldClose()) {

        draw();

        BeginDrawing();



        
        ClearBackground(GRAY);
        DrawTexture(screen.texture,0,0,WHITE);

        EndDrawing();
    }


    UnloadRenderTexture(screen);
    CloseWindow();
    return 0;
}






void draw(){
    if(IsMouseButtonDown(MOUSE_BUTTON_LEFT)){
        BeginTextureMode(screen);
        Vector2 mouse = GetMousePosition();
        mouse.y = windowSize - mouse.y;
        DrawCircleV(mouse,5,WHITE);
        
        EndTextureMode();
    }
}

float* getImagePixelData(){
    
    Image image = LoadImageFromTexture(screen.texture);
    Color *srcPixels = LoadImageColors(image);

    static float downSampledData[28*28];

    int colorChannel;

    for (size_t gridY = 0; gridY < 28; gridY++)
    {
        for (size_t gridX = 0; gridX < 28; gridX++)
        {

            colorChannel = 0;

            for (size_t srcY = 0; srcY < 16; srcY++)
            {
                for (size_t srcX = 0; srcX < 16; srcX++)
                {
                      int x = gridX * 16 + srcX;
                      int y = gridY * 16 + srcY;

                      colorChannel += srcPixels[y*448+x].r;
                }   
            }

            downSampledData[gridY * 28 + gridX] = (float)(colorChannel/256);
            
        }
        
    }

    // Free VRAM/RAM allocations
    UnloadImageColors(srcPixels);
    UnloadImage(image);

    return downSampledData;
}