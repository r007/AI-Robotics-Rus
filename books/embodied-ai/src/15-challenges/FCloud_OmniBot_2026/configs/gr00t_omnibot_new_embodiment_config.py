"""GR00T N1.7 NEW_EMBODIMENT config for OmniBot tabletop data."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig
from gr00t.data.types import ActionFormat
from gr00t.data.types import ActionRepresentation
from gr00t.data.types import ActionType
from gr00t.data.types import ModalityConfig


omnibot_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head", "hand_left", "hand_right"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["joint", "gripper"],
    ),
    "action": ModalityConfig(
        # Training trajectories should be at least 16 frames. The old 3-frame
        # smoke logs need a temporary local override if they are rechecked.
        delta_indices=list(range(0, 16)),
        modality_keys=["joint", "gripper", "base"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="joint",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(omnibot_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
