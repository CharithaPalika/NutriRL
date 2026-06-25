import torch
import torch.nn as nn


class _FoodMenuEncoder(nn.Module):
    """
    Encodes physiological state once, and each food in a (B, K, food_dim) menu
    through a SHARED per-food MLP (so the network doesn't memorise "slot 3 is
    good" — it has to actually look at the food embedding in that slot, since
    menus are resampled every step).

    Produces:
        phys_out  : (B, hidden_size2)            -- physiological encoding
        food_out  : (B, K, hidden_size2)          -- per-food encodings
        joint     : (B, K, 2*hidden_size2)        -- phys broadcast + concat per food
    """

    def __init__(self, num_states, food_embedding_size, hidden_size1=256, hidden_size2=128):
        super().__init__()
        self.physiological_head = nn.Sequential(
            nn.Linear(num_states, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
        )
        # Shared across every menu slot -> permutation-equivariant in K.
        self.food_embedding_head = nn.Sequential(
            nn.Linear(food_embedding_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
        )

    def forward(self, phys_state, food_embeddings):
        """
        phys_state      : (B, num_states)
        food_embeddings : (B, K, food_embedding_size)  OR  (B, food_embedding_size)
                          The 2D case is treated as K=1 for backward compatibility
                          with single-food observations.
        """
        if food_embeddings.dim() == 2:
            food_embeddings = food_embeddings.unsqueeze(1)  # (B, 1, food_dim)

        B, K, _ = food_embeddings.shape

        phys_out = self.physiological_head(phys_state)               # (B, H)
        food_out = self.food_embedding_head(food_embeddings)         # (B, K, H)

        phys_exp = phys_out.unsqueeze(1).expand(-1, K, -1)            # (B, K, H)
        joint    = torch.cat([phys_exp, food_out], dim=-1)            # (B, K, 2H)

        return phys_out, food_out, joint


class SharedActorCritic(nn.Module):  # SharedAC
    """
    Menu-aware shared actor-critic.

    Given the physiological state and a menu of K foods, produces:
        action_logits : (B, K+1)  -- a score per menu slot (eat slot i) plus
                                     one extra "skip" logit
        value         : (B, 1)

    The "skip" logit is computed from the physiological state alone (it
    doesn't depend on any particular food). Per-food logits come from a
    shared scoring head applied independently to each (phys, food) pair, so
    the architecture works for any num_foods (including the legacy
    single-food, K=1 case) without retraining a fixed-size action head.
    """

    def __init__(
        self,
        num_states=3,
        food_embedding_size=10,
        num_actions=2,
        hidden_size1=256,
        hidden_size2=128,
        seed=None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.encoder = _FoodMenuEncoder(num_states, food_embedding_size, hidden_size1, hidden_size2)

        # Per-food "eat this slot" score, shared across slots.
        self.food_score_head = nn.Sequential(
            nn.Linear(2 * hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1),
        )

        # "Skip" score, from physiological state alone.
        self.skip_score_head = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1),
        )

        self.val_net = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, phys_state, food_embeddings):
        phys_out, food_out, joint = self.encoder(phys_state, food_embeddings)

        food_logits = self.food_score_head(joint).squeeze(-1)     # (B, K)
        skip_logit  = self.skip_score_head(phys_out)               # (B, 1)
        action_logits = torch.cat([food_logits, skip_logit], dim=-1)  # (B, K+1)

        value = self.val_net(phys_out)  # value depends on state only, not the menu

        return action_logits, value

    def policy_parameters(self):
        return list(self.food_score_head.parameters()) + list(self.skip_score_head.parameters())

    def value_parameters(self):
        return self.val_net.parameters()


class SharedActorCriticGRU(nn.Module):  # SharedAC (GRU version)
    """
    GRU-recurrent variant, retained for sequential rollouts. Food menus are
    still scored per-slot via a shared head (see SharedActorCritic).
    """

    def __init__(
        self,
        num_states=3,
        food_embedding_size=10,
        num_actions=2,
        hidden_size1=256,
        hidden_size2=128,
        seed=None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.physiological_head = nn.GRU(
            input_size=num_states, hidden_size=hidden_size2, batch_first=True
        )
        self.food_embedding_head = nn.GRU(
            input_size=food_embedding_size, hidden_size=hidden_size2, batch_first=True
        )

        self.food_score_head = nn.Sequential(
            nn.Linear(2 * hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1),
        )
        self.skip_score_head = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1),
        )
        self.val_net = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1),
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

    def forward(self, phys_state, food_embeddings):
        # phys_state: (B, num_states) or (B, T, num_states)
        if phys_state.dim() == 2:
            phys_state = phys_state.unsqueeze(1)
        _, phys_hidden = self.physiological_head(phys_state)
        phys_out = phys_hidden[-1]  # (B, H)

        # food_embeddings: (B, K, food_dim) menu, or (B, food_dim) single food
        if food_embeddings.dim() == 2:
            food_embeddings = food_embeddings.unsqueeze(1)  # treat as K=1
        B, K, _ = food_embeddings.shape

        # Run the GRU once per menu slot (shared weights), batched over (B*K).
        flat_food = food_embeddings.reshape(B * K, 1, -1)
        _, food_hidden = self.food_embedding_head(flat_food)
        food_out = food_hidden[-1].view(B, K, -1)  # (B, K, H)

        phys_exp = phys_out.unsqueeze(1).expand(-1, K, -1)
        joint = torch.cat([phys_exp, food_out], dim=-1)  # (B, K, 2H)

        food_logits = self.food_score_head(joint).squeeze(-1)        # (B, K)
        skip_logit = self.skip_score_head(phys_out)                   # (B, 1)
        action_logits = torch.cat([food_logits, skip_logit], dim=-1)  # (B, K+1)

        value = self.val_net(phys_out)

        return action_logits, value

    def policy_parameters(self):
        return list(self.food_score_head.parameters()) + list(self.skip_score_head.parameters())

    def value_parameters(self):
        return self.val_net.parameters()


class Actor(nn.Module):
    """Menu-aware actor: scores each food slot + a skip option (see SharedActorCritic)."""

    def __init__(
        self,
        num_states=3,
        food_embedding_size=10,
        num_actions=2,
        hidden_size1=256,
        hidden_size2=128,
        seed=None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.encoder = _FoodMenuEncoder(num_states, food_embedding_size, hidden_size1, hidden_size2)

        self.food_score_head = nn.Sequential(
            nn.Linear(2 * hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1),
        )
        self.skip_score_head = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, phys_state, food_embeddings):
        phys_out, food_out, joint = self.encoder(phys_state, food_embeddings)
        food_logits = self.food_score_head(joint).squeeze(-1)
        skip_logit = self.skip_score_head(phys_out)
        action_logits = torch.cat([food_logits, skip_logit], dim=-1)
        return action_logits

    def policy_parameters(self):
        return list(self.food_score_head.parameters()) + list(self.skip_score_head.parameters())


class Critic(nn.Module):
    """
    State-value critic. Value depends only on physiological state (menus are
    resampled every step and don't define a stable state component), but the
    food menu is still accepted and pooled in for architectural symmetry with
    Actor/SharedActorCritic and to allow experimentation with menu-conditioned
    value estimates.
    """

    def __init__(
        self,
        num_states=3,
        food_embedding_size=10,
        num_actions=2,
        hidden_size1=256,
        hidden_size2=128,
        seed=None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.encoder = _FoodMenuEncoder(num_states, food_embedding_size, hidden_size1, hidden_size2)

        self.val_net = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, phys_state, food_embeddings):
        phys_out, _, _ = self.encoder(phys_state, food_embeddings)
        value = self.val_net(phys_out)
        return value

    def value_parameters(self):
        return self.val_net.parameters()


class QNetwork(nn.Module):
    """
    Menu-aware Q-network.

    forward(phys_state, food_embeddings) -> (B, K+1) Q-values, one per menu
    slot plus skip — mirroring SharedActorCritic's action_logits shape. This
    replaces the old per-action one-hot interface (which assumed a fixed,
    food-independent action space) since with a resampled menu, "action 2"
    has no fixed meaning across steps; what's fixed is "the food sitting in
    slot 2 right now", which is exactly what's fed into food_embeddings.

    DDQN/SAC agents should call this once per state to get all K+1 Q-values,
    instead of looping over a fixed action count.
    """

    def __init__(
        self,
        num_states=3,
        food_embedding_size=10,
        action_size=1,
        hidden_size1=256,
        hidden_size2=128,
        seed=None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.encoder = _FoodMenuEncoder(num_states, food_embedding_size, hidden_size1, hidden_size2)

        self.food_q_head = nn.Sequential(
            nn.Linear(2 * hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1),
        )
        self.skip_q_head = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, phys_state, food_embeddings):
        phys_out, food_out, joint = self.encoder(phys_state, food_embeddings)
        food_q = self.food_q_head(joint).squeeze(-1)   # (B, K)
        skip_q = self.skip_q_head(phys_out)              # (B, 1)
        return torch.cat([food_q, skip_q], dim=-1)        # (B, K+1)

    def Q_parameters(self):
        return list(self.food_q_head.parameters()) + list(self.skip_q_head.parameters())
