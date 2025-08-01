#include "sequence_learn.h"

int main() {
    SequenceLearn env = {.reward_rule = 1, .production_rule = 0};  // 1 = partial match (easier)
    env.observations = (float*)calloc(OBSERVATION_SIZE, sizeof(float));
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);
    
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
    
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    
    return 0;
} 