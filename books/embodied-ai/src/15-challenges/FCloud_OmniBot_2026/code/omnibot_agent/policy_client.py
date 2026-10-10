"""Small policy-client wrappers used by competition runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from omnibot_agent.pi05_adapter import EBenchActionChunk
from omnibot_agent.pi05_adapter import OmniBotObservation
from omnibot_agent.pi05_adapter import pack_ebench_request
from omnibot_agent.pi05_adapter import split_ebench_actions


class ActionPolicy(Protocol):
    def infer(self, observation: OmniBotObservation) -> EBenchActionChunk:
        ...


@dataclass
class FakePolicy:
    horizon: int = 50
    action_dim: int = 19

    def infer(self, observation: OmniBotObservation) -> EBenchActionChunk:
        del observation
        return split_ebench_actions(np.zeros((self.horizon, self.action_dim), dtype=np.float32))


@dataclass
class Pi05WebsocketPolicy:
    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self) -> None:
        from openpi_client import websocket_client_policy

        self._client = websocket_client_policy.WebsocketClientPolicy(host=self.host, port=self.port)

    def infer(self, observation: OmniBotObservation) -> EBenchActionChunk:
        response = self._client.infer(pack_ebench_request(observation))
        return split_ebench_actions(response["actions"])
