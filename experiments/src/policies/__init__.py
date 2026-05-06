"""Пакет политик рекомендаций.

Реестр четырёх активных политик основного эксперимента — в ``registry``.
Остальные модули пакета (``mmr_policy``, ``dpp_policy``, ``gnn_policy`` и т. д.)
сохраняются как legacy и в основном эксперименте не участвуют.
"""
from .registry import ACTIVE_POLICY_NAMES, active_policies

__all__ = ["ACTIVE_POLICY_NAMES", "active_policies"]
