"""Стадия 4: обучение Constrained PPO в параметрическом симуляторе с learned relevance.

Среда: gym-обёртка над симулятором. Эпизод = одна полная конференция (16 слотов)
для одного пользователя. Действие в слоте — выбор зала (среди 3-х доступных).
Reward: relevance - β * overflow_excess.

Параметр β адаптируется через Lagrangian dual:
β_{t+1} = β_t + η * (observed_overflow - target_overflow)
target_overflow = 0 (полное соблюдение).

После обучения сохраняем модель в data/models/ppo_policy.zip.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulator import Conference, UserProfile, LearnedPreferenceFn  # noqa: E402

MODEL_OUT = ROOT / "data" / "models" / "ppo_policy.zip"


class ConferenceEnv(gym.Env):
    """Gym-среда: один эпизод = одна персона проходит всю конференцию."""
    metadata = {"render_modes": []}

    def __init__(
        self,
        conf,
        personas,
        relevance_fn,
        target_overflow=0.0,
        beta_init=2.0,
        beta_lr=0.05,
    ):
        super().__init__()
        self.conf = conf
        self.personas = personas  # list of UserProfile
        self.relevance_fn = relevance_fn
        self.target_overflow = target_overflow
        self.beta = beta_init
        self.beta_lr = beta_lr

        self.halls_sorted = sorted(conf.halls.keys())
        self.n_halls = len(self.halls_sorted)
        self.n_slots = len(conf.slots)
        self.emb_dim = next(iter(conf.talks.values())).embedding.shape[0]

        # observation: persona_emb (384) + hall_load (3) + slot_one_hot (16)
        self.obs_dim = self.emb_dim + self.n_halls + self.n_slots
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(self.obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.n_halls)

        # State (per-episode)
        self.current_persona_idx = 0
        self.current_slot_idx = 0
        # Hall load (shared across episodes within a "round" — TODO).
        # Для простоты: каждый эпизод — отдельная персона на отдельной "арене".
        # Это упрощение: PPO обучается на средне-загруженной программе.
        # Глобальная нагрузка обновляется в-эпизоде.
        self.hall_load = np.zeros(self.n_halls, dtype=np.int32)
        # Episode-wide
        self.episode_overflow_count = 0
        self.episode_total_steps = 0

    def _seed_load(self, rng):
        """Заполнить начальную нагрузку залов случайным образом, имитируя midpoint."""
        # Распределение: каждый из 50-300 предыдущих участников выбрал случайный зал
        n_users = rng.integers(50, 250)
        for _ in range(n_users):
            h = rng.integers(self.n_halls)
            self.hall_load[h] += 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        # Случайная персона
        self.current_persona_idx = rng.integers(len(self.personas))
        self.current_slot_idx = 0
        self.hall_load = np.zeros(self.n_halls, dtype=np.int32)
        self.episode_overflow_count = 0
        self.episode_total_steps = 0
        return self._obs(), {}

    def _obs(self):
        persona = self.personas[self.current_persona_idx]
        hall_fractions = np.zeros(self.n_halls, dtype=np.float32)
        # Если эпизод закончился, возвращаем последний слот для obs (не используется PPO для done=True)
        slot_idx = min(self.current_slot_idx, self.n_slots - 1)
        slot = self.conf.slots[slot_idx]
        for i, hid in enumerate(self.halls_sorted):
            cap = self.conf.halls[hid].capacity
            hall_fractions[i] = float(self.hall_load[i] / max(1.0, cap))
        slot_one_hot = np.zeros(self.n_slots, dtype=np.float32)
        slot_one_hot[slot_idx] = 1.0
        return np.concatenate([
            persona.embedding.astype(np.float32),
            hall_fractions,
            slot_one_hot,
        ])

    def action_masks(self):
        """Mask: 1 если зал имеет доклад в текущем слоте И не переполнен."""
        mask = np.zeros(self.n_halls, dtype=bool)
        slot_idx = min(self.current_slot_idx, self.n_slots - 1)
        slot = self.conf.slots[slot_idx]
        halls_in_slot = {self.conf.talks[tid].hall for tid in slot.talk_ids}
        for i, hid in enumerate(self.halls_sorted):
            cap = self.conf.halls[hid].capacity
            occ = self.hall_load[i]
            if hid in halls_in_slot and occ < cap * 0.95:
                mask[i] = True
        if not mask.any():
            # все переполнены или нет докладов — разрешаем все из существующих в слоте
            for i, hid in enumerate(self.halls_sorted):
                if hid in halls_in_slot:
                    mask[i] = True
        if not mask.any():
            mask = np.ones(self.n_halls, dtype=bool)
        return mask

    def step(self, action):
        slot = self.conf.slots[self.current_slot_idx]
        chosen_hall = self.halls_sorted[int(action)]

        # Найти доклад в выбранном зале в этом слоте
        chosen_talk = None
        for tid in slot.talk_ids:
            if self.conf.talks[tid].hall == chosen_hall:
                chosen_talk = tid
                break

        if chosen_talk is None:
            # выбрали зал без доклада — отрицательная награда
            reward = -1.0
        else:
            persona = self.personas[self.current_persona_idx]
            talk = self.conf.talks[chosen_talk]
            relevance = float(self.relevance_fn(persona.embedding, talk.embedding))
            cap = self.conf.halls[chosen_hall].capacity
            occ_before = self.hall_load[int(action)]
            overflow_excess = max(0.0, (occ_before + 1 - cap) / cap)
            if occ_before >= cap:
                self.episode_overflow_count += 1
            reward = relevance - self.beta * overflow_excess
            # Update load
            self.hall_load[int(action)] += 1

        self.episode_total_steps += 1
        self.current_slot_idx += 1
        terminated = self.current_slot_idx >= self.n_slots
        truncated = False
        info = {}
        if terminated:
            # Lagrangian update
            of_rate = self.episode_overflow_count / max(1, self.episode_total_steps)
            self.beta = max(0.1, self.beta + self.beta_lr * (of_rate - self.target_overflow))
            info["episode_overflow_rate"] = of_rate
            info["beta"] = self.beta
        else:
            # Сбросить hall_load для следующего слота — каждый слот независимый,
            # нагрузка не переносится (это была семантическая ошибка)
            rng = np.random.default_rng(self.current_slot_idx * 1000 + self.current_persona_idx)
            self.hall_load = np.zeros(self.n_halls, dtype=np.int32)
            self._seed_load(rng)
        return self._obs(), float(reward), terminated, truncated, info


def text_of(p):
    return p.get("background") or p.get("profile") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-timesteps", type=int, default=100_000)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    conf = Conference.load(
        ROOT / "data" / "conferences" / "mobius_2025_autumn.json",
        ROOT / "data" / "conferences" / "mobius_2025_autumn_embeddings.npz",
    )

    with open(ROOT / "data" / "personas" / "personas.json", encoding="utf-8") as f:
        personas_meta = json.load(f)
    npz = np.load(ROOT / "data" / "personas" / "personas_embeddings.npz", allow_pickle=False)
    by_id = {pid: npz["embeddings"][i] for i, pid in enumerate(npz["ids"])}
    user_profiles = [
        UserProfile(id=p["id"], text=text_of(p), embedding=by_id[p["id"]])
        for p in personas_meta
    ]

    relevance_fn = LearnedPreferenceFn(ROOT / "data" / "models" / "preference_model.pkl")
    # precompute
    persona_dict = {u.id: u.embedding for u in user_profiles}
    talk_dict = {tid: t.embedding for tid, t in conf.talks.items()}
    relevance_fn.precompute_all(persona_dict, talk_dict)
    print(f"Precomputed {len(relevance_fn._cache)} preference values")

    def make_env(seed):
        env = ConferenceEnv(
            conf=conf,
            personas=user_profiles,
            relevance_fn=relevance_fn,
            target_overflow=0.0,
            beta_init=2.0,
            beta_lr=0.05,
        )
        env.reset(seed=seed)
        env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
        return env

    # Single env (не parallelize, простой режим)
    env = make_env(args.seed)
    print(f"Env obs_dim: {env.unwrapped.obs_dim}, action_dim: {env.unwrapped.n_halls}")

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        seed=args.seed,
        policy_kwargs={"net_arch": [128, 64]},
    )
    print(f"Training MaskablePPO for {args.total_timesteps} steps...")
    model.learn(total_timesteps=args.total_timesteps)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT)
    print(f"WROTE: {MODEL_OUT}")

    # Quick evaluation
    print("\nEvaluating...")
    n_eval = 100
    overflow_rates = []
    rewards = []
    for ep in range(n_eval):
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
    print(f"Eval over {n_eval} eps:")
    print(f"  Mean reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"  Mean overflow rate: {np.mean(overflow_rates):.3f}")


if __name__ == "__main__":
    main()
