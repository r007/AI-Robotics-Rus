import json

import numpy as np
import pytest

from lightnav_lesson import decode_response


def response(actions=None, sequence=1):
    if actions is None:
        actions = np.zeros((10, 3)).tolist()
    return json.dumps({"action": "next", "data": {"rc": 0, "seq": sequence,
                       "stop": False, "actions": {"actions": actions}}})


def test_official_nested_actions():
    data, path = decode_response(response(), 1)
    assert path.shape == (10, 3)
    assert data["stop"] is False


def test_sequence_mismatch():
    with pytest.raises(RuntimeError, match="Invalid inference"):
        decode_response(response(sequence=2), 1)


@pytest.mark.parametrize("actions", [[[0, 0, 0]], np.full((10, 3), np.nan).tolist(), []])
def test_invalid_trajectory(actions):
    with pytest.raises(RuntimeError, match="finite 10x3"):
        decode_response(response(actions=actions), 1)


def test_no_stop_decision():
    message = json.loads(response())
    message["data"].pop("stop")
    with pytest.raises(RuntimeError, match="Boolean"):
        decode_response(json.dumps(message), 1)
