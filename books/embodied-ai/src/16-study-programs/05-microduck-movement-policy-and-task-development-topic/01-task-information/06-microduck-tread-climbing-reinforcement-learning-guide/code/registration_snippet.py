"""Task registration sketch for microduck_rl/mjlab task registry.

Paste the registration block into the package's ``tasks/__init__.py``.  The
existing ``MicroduckOnPolicyRunner`` class is defined in that module, so it is
intentionally not imported from a fictitious helper module here.
"""

from mjlab.tasks.registry import register_mjlab_task

from .microduck_ladder_env_cfg import (
    MicroduckLadderRlCfg,
    make_microduck_ladder_env_cfg,
)
from .microduck_video_ladder_env_cfg import (
    MicroduckVideoLadderRlCfg,
    MicroduckVideoLadderSupportRlCfg,
    make_microduck_video_ladder_env_cfg,
    make_microduck_video_ladder_support_env_cfg,
)
from .microduck_video_ladder_footstep_env_cfg import (
    MicroduckVideoLadderFootstepRlCfg,
    make_microduck_video_ladder_footstep_env_cfg,
)


register_mjlab_task(
    task_id="Mjlab-Ladder-Climb-MicroDuck",
    env_cfg=make_microduck_ladder_env_cfg(),
    play_env_cfg=make_microduck_ladder_env_cfg(play=True),
    rl_cfg=MicroduckLadderRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Video-Ladder-Climb-MicroDuck",
    env_cfg=make_microduck_video_ladder_env_cfg(),
    play_env_cfg=make_microduck_video_ladder_env_cfg(play=True),
    rl_cfg=MicroduckVideoLadderRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Video-Ladder-Support-Stage-MicroDuck",
    env_cfg=make_microduck_video_ladder_support_env_cfg(),
    play_env_cfg=make_microduck_video_ladder_support_env_cfg(play=True),
    rl_cfg=MicroduckVideoLadderSupportRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Video-Ladder-Footstep-MicroDuck",
    env_cfg=make_microduck_video_ladder_footstep_env_cfg(),
    play_env_cfg=make_microduck_video_ladder_footstep_env_cfg(play=True),
    rl_cfg=MicroduckVideoLadderFootstepRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
