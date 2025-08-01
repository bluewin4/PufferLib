#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_GREEN = (Color){0, 187, 0, 255};
const Color PUFF_BLUE = (Color){0, 0, 187, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

// Configuration constants
#define SEQUENCE_LENGTH 4
#define TRIALS_PER_EPISODE 16
#define STEPS_PER_EPISODE 64
#define NUM_VALUES 4  // Dimensionality of values (0, 1, 2, 3)
#define OBSERVATION_SIZE (SEQUENCE_LENGTH + 2)  // Current sequence + trial number + step in trial

// Only use floats!
typedef struct {
    float score;
    float trials_to_max_reward;
    float total_reward;
    float current_trial_reward;
    float n; // Required as the last field 
} Log;

typedef struct {
    Log log;                     // Required field
    float* observations;         // Required field. Ensure type matches in .py and .c
    int* actions;                // Required field. Ensure type matches in .py and .c
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field
    
    // Environment specific fields
    int target_sequence[SEQUENCE_LENGTH];
    int current_sequence[SEQUENCE_LENGTH];
    int sequence_position;
    int current_trial;
    int episode_step;
    int steps_in_trial;
    int max_reward_achieved;
    float reward_rule;  // 0: exact match, 1: partial match, 2: exponential
    int production_rule;  // 0: flat (all available), 1: associative
} SequenceLearn;

// Function declarations
void generate_target_sequence(SequenceLearn* env);
float calculate_reward(SequenceLearn* env);
void update_observations(SequenceLearn* env);
int is_action_valid(SequenceLearn* env, int action);

void c_reset(SequenceLearn* env) {
    // Generate new target sequence for the episode
    generate_target_sequence(env);
    
    // Reset counters
    env->current_trial = 0;
    env->episode_step = 0;
    env->steps_in_trial = 0;
    env->sequence_position = 0;
    env->max_reward_achieved = 0;
    env->log.total_reward = 0;
    env->log.trials_to_max_reward = -1;
    
    // Clear current sequence
    memset(env->current_sequence, -1, sizeof(env->current_sequence));
    
    // Update observations
    update_observations(env);
}

void c_step(SequenceLearn* env) {
    env->rewards[0] = 0;
    env->terminals[0] = 0;
    
    // Check if action is valid based on production rule
    if (is_action_valid(env, env->actions[0])) {
        // Add action to current sequence
        env->current_sequence[env->sequence_position] = env->actions[0];
        env->sequence_position++;
        env->steps_in_trial++;
    }
    
    env->episode_step++;
    
    // Check if we've completed a sequence
    if (env->sequence_position >= SEQUENCE_LENGTH) {
        // Calculate reward for this trial
        float trial_reward = calculate_reward(env);
        env->rewards[0] = trial_reward;
        env->log.current_trial_reward = trial_reward;
        env->log.total_reward += trial_reward;
        
        // Check if we achieved max reward for the first time
        if (trial_reward >= 1.0f && !env->max_reward_achieved) {
            env->max_reward_achieved = 1;
            env->log.trials_to_max_reward = env->current_trial + 1;
        }
        
        // Reset for next trial
        env->current_trial++;
        env->sequence_position = 0;
        env->steps_in_trial = 0;
        memset(env->current_sequence, -1, sizeof(env->current_sequence));
        
        // Check if episode is complete
        if (env->current_trial >= TRIALS_PER_EPISODE || env->episode_step >= STEPS_PER_EPISODE) {
            env->terminals[0] = 1;
            env->log.score = env->log.total_reward / TRIALS_PER_EPISODE;
            env->log.n += 1;
        }
    }
    
    // Update observations
    update_observations(env);
}

void generate_target_sequence(SequenceLearn* env) {
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
        env->target_sequence[i] = rand() % NUM_VALUES;
    }
}

float calculate_reward(SequenceLearn* env) {
    float reward = 0.0f;
    
    if (env->reward_rule == 0) {  // Exact match
        int matches = 1;
        for (int i = 0; i < SEQUENCE_LENGTH; i++) {
            if (env->current_sequence[i] != env->target_sequence[i]) {
                matches = 0;
                break;
            }
        }
        reward = matches ? 1.0f : 0.0f;
    }
    else if (env->reward_rule == 1) {  // Partial match
        int matches = 0;
        for (int i = 0; i < SEQUENCE_LENGTH; i++) {
            if (env->current_sequence[i] == env->target_sequence[i]) {
                matches++;
            }
        }
        reward = (float)matches / SEQUENCE_LENGTH;
    }
    else if (env->reward_rule == 2) {  // Exponential reward
        int matches = 0;
        for (int i = 0; i < SEQUENCE_LENGTH; i++) {
            if (env->current_sequence[i] == env->target_sequence[i]) {
                matches++;
            }
        }
        reward = powf(2.0f, matches) / powf(2.0f, SEQUENCE_LENGTH);
    }
    
    return reward;
}

void update_observations(SequenceLearn* env) {
    // First SEQUENCE_LENGTH values: current sequence (-1 if not filled)
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
        env->observations[i] = (env->current_sequence[i] >= 0) ? 
            (float)env->current_sequence[i] / (NUM_VALUES - 1) : -1.0f;
    }
    
    // Trial number (normalized)
    env->observations[SEQUENCE_LENGTH] = (float)env->current_trial / TRIALS_PER_EPISODE;
    
    // Position in current sequence (normalized)
    env->observations[SEQUENCE_LENGTH + 1] = (float)env->sequence_position / SEQUENCE_LENGTH;
}

int is_action_valid(SequenceLearn* env, int action) {
    if (action < 0 || action >= NUM_VALUES) return 0;
    
    if (env->production_rule == 0) {  // Flat - all actions always valid
        return 1;
    }
    else if (env->production_rule == 1) {  // Associative - depends on previous action
        if (env->sequence_position == 0) {
            return 1;  // First action is always valid
        }
        
        int prev_action = env->current_sequence[env->sequence_position - 1];
        // Simple associative rule: action must differ by at most 1 from previous
        return abs(action - prev_action) <= 1;
    }
    
    return 1;
}

void c_render(SequenceLearn* env) {
    if (!IsWindowReady()) {
        InitWindow(960, 600, "PufferLib Sequence Learning");
        SetTargetFPS(10);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    
    // Draw target sequence
    DrawText("Target Sequence:", 20, 20, 20, PUFF_WHITE);
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
        Color color = (Color[]){PUFF_RED, PUFF_GREEN, PUFF_BLUE, PUFF_CYAN}[env->target_sequence[i]];
        DrawRectangle(200 + i * 60, 20, 50, 50, color);
        DrawText(TextFormat("%d", env->target_sequence[i]), 220 + i * 60, 35, 20, PUFF_WHITE);
    }
    
    // Draw current sequence
    DrawText("Current Sequence:", 20, 100, 20, PUFF_WHITE);
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
        if (env->current_sequence[i] >= 0) {
            Color color = (Color[]){PUFF_RED, PUFF_GREEN, PUFF_BLUE, PUFF_CYAN}[env->current_sequence[i]];
            DrawRectangle(200 + i * 60, 100, 50, 50, color);
            DrawText(TextFormat("%d", env->current_sequence[i]), 220 + i * 60, 115, 20, PUFF_WHITE);
        } else {
            DrawRectangle(200 + i * 60, 100, 50, 50, DARKGRAY);
            DrawText("?", 220 + i * 60, 115, 20, PUFF_WHITE);
        }
    }
    
    // Draw stats
    DrawText(TextFormat("Trial: %d/%d", env->current_trial + 1, TRIALS_PER_EPISODE), 20, 180, 20, PUFF_WHITE);
    DrawText(TextFormat("Episode Step: %d/%d", env->episode_step, STEPS_PER_EPISODE), 20, 210, 20, PUFF_WHITE);
    DrawText(TextFormat("Total Reward: %.2f", env->log.total_reward), 20, 240, 20, PUFF_WHITE);
    DrawText(TextFormat("Last Trial Reward: %.2f", env->log.current_trial_reward), 20, 270, 20, PUFF_WHITE);
    
    if (env->log.trials_to_max_reward > 0) {
        DrawText(TextFormat("Trials to Max Reward: %.0f", env->log.trials_to_max_reward), 20, 300, 20, GREEN);
    }
    
    EndDrawing();
}

void c_close(SequenceLearn* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
} 