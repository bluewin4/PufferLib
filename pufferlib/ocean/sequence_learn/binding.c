#include "sequence_learn.h"

#define Env SequenceLearn
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->reward_rule = unpack(kwargs, "reward_rule");
    env->production_rule = unpack(kwargs, "production_rule");
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "trials_to_max_reward", log->trials_to_max_reward);
    assign_to_dict(dict, "total_reward", log->total_reward);
    assign_to_dict(dict, "current_trial_reward", log->current_trial_reward);
    assign_to_dict(dict, "achieved_max_reward", log->achieved_max_reward);
    return 0;
} 