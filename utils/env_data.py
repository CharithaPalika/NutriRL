from collections import namedtuple

EnvData = namedtuple(
    'env_data',
    [
        'curr_phy_state',
        'curr_food_state',
        'action',
        'reward',
        'next_phy_state',
        'next_food_state',
        'action_logits',
        'log_probs',
        'value',
        'q1_val',
        'q2_val',
        'done'
    ]
)
