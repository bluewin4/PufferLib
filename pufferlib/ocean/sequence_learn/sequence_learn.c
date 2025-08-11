#include "sequence_learn.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
    #if !PFL_HAS_RAYLIB
    fprintf(stderr, "sequence_learn: raylib not available at build or lint time; standalone UI disabled.\n");
    return 0;
    #endif
    // Parse command line arguments
    int reward_rule = 3;  // Default to dense rewards
    int production_rule = 0;
    
    if (argc > 1) {
        reward_rule = atoi(argv[1]);
        printf("Using reward_rule: %d (0=exact, 1=partial, 2=exponential, 3=dense)\n", reward_rule);
    }
    if (argc > 2) {
        production_rule = atoi(argv[2]);
        printf("Using production_rule: %d (0=flat, 1=associative)\n", production_rule);
    }
    
    printf("Sequence Length: %d, Num Values: %d\n", SEQUENCE_LENGTH, NUM_VALUES);
    printf("Press 1-%d to input actions, or let it play randomly\n", NUM_VALUES);
    printf("Press ESC to exit\n\n");
    
    SequenceLearn env = {.reward_rule = reward_rule, .production_rule = production_rule};
    env.observations = (float*)calloc(OBSERVATION_SIZE, sizeof(float));
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);

    #if PFL_HAS_RAYLIB
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_ONE)) {
                env.actions[0] = 0;
            } else if (IsKeyDown(KEY_TWO)) {
                env.actions[0] = 1;
            } else if (IsKeyDown(KEY_THREE)) {
                env.actions[0] = 2;
            } else if (IsKeyDown(KEY_FOUR)) {
                env.actions[0] = 3;
            } else {
                env.actions[0] = -1;
            }
        } else {
            env.actions[0] = rand() % NUM_VALUES;
        }
        
        c_step(&env);
        
        // Check if episode ended and reset
        if (env.terminals[0]) {
            c_reset(&env);
        }
        
        c_render(&env);
        
        // Small delay to make it playable
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            WaitTime(0.2);
        }
    }
    #else
    for (int t = 0; t < 1000; t++) {
        env.actions[0] = rand() % NUM_VALUES;
        c_step(&env);
        if (env.terminals[0]) {
            c_reset(&env);
        }
    }
    #endif
    
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    
    return 0;
} 