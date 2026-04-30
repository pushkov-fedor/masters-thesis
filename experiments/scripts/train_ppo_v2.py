"""Multi-agent batch PPO в правильно поставленной congestion game.

Эпизод = вся конференция (16 слотов Mobius).
Внутри слота: батч пользователей обрабатывается ПОСЛЕДОВАТЕЛЬНО, каждый
последующий видит обновлённый hall_load.

State (observation):
  persona_emb (384) ⊕ hall_loads (n_halls) ⊕ slot_one_hot (n_slots)
  ⊕ users_remaining_in_slot (1) ⊕ fame_in_slot (3) = 384+3+16+1+3=407

Action: дискретный — выбрать зал для текущего пользователя из доступных в слоте.
Action mask: исключаем переполненные залы И залы без доклада в слоте.

Reward за каждый шаг:
  relevance(persona, talk_in_chosen_hall)  +  w_fame·fame
  - β·overflow_excess_after_step

β адаптируется через Lagrangian dual в конце эпизода.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulator import Conference, UserProfile, LearnedPreferenceFn  # noqa: E402

MODEL_OUT = ROOT / "data" / "models" / "ppo_v2_policy.zip"


class MultiAgentConferenceEnv(gym.Env):
    """Эпизод = одна полная конференция, batch users в каждом слоте.

    Каждый шаг = одна рекомендация одному пользователю; обновление hall_load
    после каждого шага создаёт реальный congestion-эффект.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        conf,
        personas,
        relevance_fn,
        batch_size_per_slot: int = 50,
        beta_init: float = 2.0,
        beta_lr: float = 0.05,
        target_overflow: float = 0.0,
        w_fame: float = 0.3,
    ):
        super().__init__()
        self.conf = conf
        self.personas = personas
        self.relevance_fn = relevance_fn
        self.batch_size_per_slot = batch_size_per_slot
        self.beta = beta_init
        self.beta_lr = beta_lr
        self.target_overflow = target_overflow
        self.w_fame = w_fame

        self.halls_sorted = sorted(conf.halls.keys())
        self.n_halls = len(self.halls_sorted)
        self.n_slots = len(conf.slots)
        self.emb_dim = next(iter(conf.talks.values())).embedding.shape[0]

        # observation: persona_emb + hall_loads + slot_one_hot + users_remain + fame_in_slot
        self.obs_dim = self.emb_dim + self.n_halls + self.n_slots + 1 + self.n_halls
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(self.obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.n_halls)

        # Episode state
        self.current_slot_idx = 0
        self.user_idx_in_slot = 0
        self.batch_users = []  # текущий батч пользователей в слоте
        self.hall_load_per_slot = {}  # {(slot_id, hall_id): count}
        self.episode_overflow_count = 0
        self.episode_total_steps = 0

    def _sample_batch(self, rng):
        """Случайный батч пользователей для слота."""
        idx = rng.choice(len(self.personas), size=self.batch_size_per_slot, replace=True)
        return [self.personas[i] for i in idx]

    def _new_slot(self, rng):
        self.batch_users = self._sample_batch(rng)
        self.user_idx_in_slot = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.current_slot_idx = 0
        self.episode_overflow_count = 0
        self.episode_total_steps = 0
        self.hall_load_per_slot = {}
        for s in self.conf.slots:
            for hid in self.halls_sorted:
                self.hall_load_per_slot[(s.id, hid)] = 0
        self._rng = rng
        # Поиск первого слота с докладами
        while (self.current_slot_idx < self.n_slots and
               not self.conf.slots[self.current_slot_idx].talk_ids):
            self.current_slot_idx += 1
        if self.current_slot_idx < self.n_slots:
            self._new_slot(rng)
        return self._obs(), {}

    def _current_slot(self):
        if self.current_slot_idx >= self.n_slots:
            return self.conf.slots[-1]
        return self.conf.slots[self.current_slot_idx]

    def _current_user(self):
        if self.user_idx_in_slot < len(self.batch_users):
            return self.batch_users[self.user_idx_in_slot]
        return self.batch_users[-1]

    def _obs(self):
        slot = self._current_slot()
        user = self._current_user()
        # hall fractions
        hall_fractions = np.zeros(self.n_halls, dtype=np.float32)
        for i, hid in enumerate(self.halls_sorted):
            cap = self.conf.halls[hid].capacity
            occ = self.hall_load_per_slot[(slot.id, hid)]
            hall_fractions[i] = float(occ / max(1.0, cap))
        slot_one_hot = np.zeros(self.n_slots, dtype=np.float32)
        slot_one_hot[min(self.current_slot_idx, self.n_slots - 1)] = 1.0
        users_remain = np.array([
            (self.batch_size_per_slot - self.user_idx_in_slot) / self.batch_size_per_slot
        ], dtype=np.float32)
        # fame in slot per hall
        fame_in_slot = np.zeros(self.n_halls, dtype=np.float32)
        for tid in slot.talk_ids:
            t = self.conf.talks[tid]
            hi = self.halls_sorted.index(t.hall) if t.hall in self.halls_sorted else 0
            fame_in_slot[hi] = max(fame_in_slot[hi], t.fame)
        return np.concatenate([
            user.embedding.astype(np.float32),
            hall_fractions,
            slot_one_hot,
            users_remain,
            fame_in_slot,
        ])

    def action_masks(self):
        """Mask: только залы с докладом в слоте, и не переполненные."""
        mask = np.zeros(self.n_halls, dtype=bool)
        slot = self._current_slot()
        halls_with_talks = {self.conf.talks[tid].hall for tid in slot.talk_ids}
        for i, hid in enumerate(self.halls_sorted):
            cap = self.conf.halls[hid].capacity
            occ = self.hall_load_per_slot[(slot.id, hid)]
            if hid in halls_with_talks and occ < cap * 0.95:
                mask[i] = True
        if not mask.any():
            # Все переполнены — маска включает залы с докладами
            for i, hid in enumerate(self.halls_sorted):
                if hid in halls_with_talks:
                    mask[i] = True
        if not mask.any():
            mask = np.ones(self.n_halls, dtype=bool)
        return mask

    def step(self, action):
        slot = self._current_slot()
        user = self._current_user()
        chosen_hall = self.halls_sorted[int(action)]

        chosen_talk = None
        for tid in slot.talk_ids:
            if self.conf.talks[tid].hall == chosen_hall:
                chosen_talk = tid
                break

        if chosen_talk is None:
            reward = -1.0
        else:
            talk = self.conf.talks[chosen_talk]
            relevance = float(self.relevance_fn(user.embedding, talk.embedding))
            effective = (1 - self.w_fame) * relevance + self.w_fame * talk.fame
            cap = self.conf.halls[chosen_hall].capacity
            occ_before = self.hall_load_per_slot[(slot.id, chosen_hall)]
            overflow_excess = max(0.0, (occ_before + 1 - cap) / cap)
            if occ_before >= cap:
                self.episode_overflow_count += 1
            reward = effective - self.beta * overflow_excess
            self.hall_load_per_slot[(slot.id, chosen_hall)] += 1

        self.episode_total_steps += 1
        self.user_idx_in_slot += 1

        # Перейти к следующему пользователю / слоту
        if self.user_idx_in_slot >= len(self.batch_users):
            # Все пользователи слота обработаны → переход к следующему слоту
            self.current_slot_idx += 1
            while (self.current_slot_idx < self.n_slots and
                   not self.conf.slots[self.current_slot_idx].talk_ids):
                self.current_slot_idx += 1
            if self.current_slot_idx < self.n_slots:
                self._new_slot(self._rng)

        terminated = self.current_slot_idx >= self.n_slots
        truncated = False
        info = {}
        if terminated:
            of_rate = self.episode_overflow_count / max(1, self.episode_total_steps)
            self.beta = max(0.1, self.beta + self.beta_lr * (of_rate - self.target_overflow))
            info["episode_overflow_rate"] = of_rate
            info["beta"] = self.beta
            info["episode_total_steps"] = self.episode_total_steps
        return self._obs(), float(reward), terminated, truncated, info


def text_of(p):
    return p.get("background") or p.get("profile") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-timesteps", type=int, default=2_000_000)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size-per-slot", type=int, default=50)
    ap.add_argument("--w-fame", type=float, default=0.3)
    args = ap.parse_args()

    conf = Conference.load(
        ROOT / "data" / "conferences" / "mobius_2025_autumn.json",
        ROOT / "data" / "conferences" / "mobius_2025_autumn_embeddings.npz",
    )

    with open(ROOT / "data" / "personas" / "personas.json", encoding="utf-8") as f:
        personas_meta = json.load(f)
    npz = np.load(ROOT / "data" / "personas" / "personas_embeddings.npz", allow_pickle=False)
    by_id = {pid: npz["embeddings"][i] for i, pid in enumerate(npz["ids"])}
    personas = [
        UserProfile(id=p["id"], text=text_of(p), embedding=by_id[p["id"]])
        for p in personas_meta
    ]

    relevance_fn = LearnedPreferenceFn(ROOT / "data" / "models" / "preference_model.pkl")
    persona_dict = {u.id: u.embedding for u in personas}
    talk_dict = {tid: t.embedding for tid, t in conf.talks.items()}
    relevance_fn.precompute_all(persona_dict, talk_dict)
    print(f"Precomputed {len(relevance_fn._cache)} preferences")

    env = MultiAgentConferenceEnv(
        conf=conf, personas=personas, relevance_fn=relevance_fn,
        batch_size_per_slot=args.batch_size_per_slot,
        w_fame=args.w_fame,
    )
    env.reset(seed=args.seed)
    env = ActionMasker(env, lambda e: e.unwrapped.action_masks())

    print(f"obs_dim={env.unwrapped.obs_dim}, action_dim={env.unwrapped.n_halls}")
    print(f"Episode steps: {env.unwrapped.batch_size_per_slot} × ~{env.unwrapped.n_slots-4} multi-talk slots = {args.batch_size_per_slot * (env.unwrapped.n_slots-4)}")

    model = MaskablePPO(
        "MlpPolicy", env,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        seed=args.seed,
        policy_kwargs={"net_arch": [256, 128]},
    )
    print(f"Training for {args.total_timesteps} steps...")
    model.learn(total_timesteps=args.total_timesteps)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT)
    print(f"WROTE: {MODEL_OUT}")

    # Eval
    print("\nEvaluating on 50 episodes...")
    overflow_rates = []
    rewards = []
    for ep in range(50):
        obs, _ = env.reset(seed=1000 + ep)
        ep_reward = 0
        done = False
        while not done:
            mask = env.unwrapped.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            if done:
                overflow_rates.append(info.get("episode_overflow_rate", 0))
        rewards.append(ep_reward)
    print(f"Eval over 50 eps:")
    print(f"  Mean reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"  Mean overflow rate: {np.mean(overflow_rates):.4f}")


if __name__ == "__main__":
    main()
