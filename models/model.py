import torch
import torch.nn as nn
# 
# 
class SharedActorCritic(nn.Module): # SharedAC
    def __init__(
        self,
        num_states=3,
        food_embedding_size=10,
        num_actions=2,
        hidden_size1=256,
        hidden_size2=128, seed = None
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.physiological_head = nn.Sequential(
            nn.Linear(num_states, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU()
        )

        self.food_embedding_head = nn.Sequential(
            nn.Linear(food_embedding_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU()
        )

        # Actor outputs logits
        self.actor = nn.Sequential(
            nn.Linear(2 * hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, num_actions)
        )

        self.val_net = nn.Sequential(
            nn.Linear(2 * hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, phys_state, food_embedding):
        phys_out = self.physiological_head(phys_state)
        food_out = self.food_embedding_head(food_embedding)

        concat = torch.cat((phys_out, food_out), dim=1)

        action_logits = self.actor(concat)
        value = self.val_net(concat)

        return action_logits, value

    def policy_parameters(self):
        return self.actor.parameters()

    def value_parameters(self):
        return self.val_net.parameters()
# 

class SharedActorCriticGRU(nn.Module):  # SharedAC (GRU version)
    def __init__(
        self,
        num_states=3,
        food_embedding_size=10,
        num_actions=2,
        hidden_size1=256,
        hidden_size2=128,
        seed=None
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        # Physiological encoder (GRU-based)
        self.physiological_head = nn.GRU(
            input_size=num_states,
            hidden_size=hidden_size2,
            batch_first=True
        )

        # Food embedding encoder (GRU-based)
        self.food_embedding_head = nn.GRU(
            input_size=food_embedding_size,
            hidden_size=hidden_size2,
            batch_first=True
        )

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(2 * hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, num_actions)
        )

        # Critic
        self.val_net = nn.Sequential(
            nn.Linear(2 * hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0)

    def forward(self, phys_state, food_embedding):
        # Ensure 3D input: (batch, seq_len, features)
        if phys_state.dim() == 2:
            phys_state = phys_state.unsqueeze(1)

        if food_embedding.dim() == 2:
            food_embedding = food_embedding.unsqueeze(1)

        # GRU forward
        _, phys_hidden = self.physiological_head(phys_state)
        _, food_hidden = self.food_embedding_head(food_embedding)

        # Take last hidden state
        phys_out = phys_hidden[-1]
        food_out = food_hidden[-1]

        concat = torch.cat((phys_out, food_out), dim=1)

        action_logits = self.actor(concat)
        value = self.val_net(concat)

        return action_logits, value

    def policy_parameters(self):
        return self.actor.parameters()

    def value_parameters(self):
        return self.val_net.parameters()




class Actor(nn.Module):
     def __init__(
         self,
         num_states=3,
         food_embedding_size=10,
         num_actions=2,
         hidden_size1=256,
         hidden_size2=128, seed = None
     ):
         super().__init__()
         if seed is not None:
             torch.manual_seed(seed)
         self.physiological_head = nn.Sequential(
             nn.Linear(num_states, hidden_size1),
             nn.ReLU(),
             nn.Linear(hidden_size1, hidden_size2),
             nn.ReLU()
         )
         self.food_embedding_head = nn.Sequential(
             nn.Linear(food_embedding_size, hidden_size1),
             nn.ReLU(),
             nn.Linear(hidden_size1, hidden_size2),
             nn.ReLU()
         )
         # Actor outputs logits
         self.actor = nn.Sequential(
             nn.Linear(2 * hidden_size2, hidden_size2),
             nn.ReLU(),
             nn.Linear(hidden_size2, num_actions)
         )
         self._init_weights()

     def _init_weights(self):
         for m in self.modules():
             if isinstance(m, nn.Linear):
                 nn.init.orthogonal_(m.weight)
                 nn.init.constant_(m.bias, 0)

     def forward(self, phys_state, food_embedding):
         phys_out = self.physiological_head(phys_state)
         food_out = self.food_embedding_head(food_embedding)
         concat = torch.cat((phys_out, food_out), dim=1)
         action_logits = self.actor(concat)
         return action_logits
     
     def policy_parameters(self):
         return self.actor.parameters()


class Critic(nn.Module):
    def __init__(
        self,
        num_states=3,
        food_embedding_size=10,
        num_actions=2,
        hidden_size1=256,
        hidden_size2=128, seed = None
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.physiological_head = nn.Sequential(
            nn.Linear(num_states, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU()
        )

        self.food_embedding_head = nn.Sequential(
            nn.Linear(food_embedding_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU()
        )

        self.val_net = nn.Sequential(
            nn.Linear(2 * hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, phys_state, food_embedding):
        phys_out = self.physiological_head(phys_state)
        food_out = self.food_embedding_head(food_embedding)

        concat = torch.cat((phys_out, food_out), dim=1)

        value = self.val_net(concat)

        return value

    def value_parameters(self):
        return self.val_net.parameters()
    
class QNetwork(nn.Module):
    def __init__(
        self,
        num_states=3,
        food_embedding_size=10,
        action_size=1,
        hidden_size1=256,
        hidden_size2=128,
        seed = None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.physiological_head = nn.Sequential(
            nn.Linear(num_states, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU()
        )

        self.food_embedding_head = nn.Sequential(
            nn.Linear(food_embedding_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU()
        )

        self.action_head = nn.Sequential(
            nn.Linear(action_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU())

        self.Q_net = nn.Sequential(
            nn.Linear(3 * hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, phys_state, food_embedding, action):
        phys_out = self.physiological_head(phys_state)
        food_out = self.food_embedding_head(food_embedding)
        action_out = self.action_head(action)
        concat = torch.cat((phys_out, food_out, action_out), dim=1)

        Qvalue = self.Q_net(concat)

        return Qvalue


    def Q_parameters(self):
        return self.Q_net.parameters()