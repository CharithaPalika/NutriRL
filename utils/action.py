from torch.distributions import Categorical

def choose_action(logits):
    dist = Categorical(logits=logits)
    action = dist.sample()
    return action, dist.log_prob(action)
