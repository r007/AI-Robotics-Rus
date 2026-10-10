"""Official LeRobot training with fail-fast pi05 pretrained-weight loading.

LeRobot 0.6.1 catches weight-loading exceptions and returns a model anyway.
For this pinned release, require its explicit successful strict-load diagnostic.
"""

import contextlib
import functools
import io


def install_strict_loader():
    from pi05_omy_utils import check_version
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    check_version()
    original = PI05Policy.from_pretrained.__func__
    if getattr(original, "_omy_strict", False):
        return

    @functools.wraps(original)
    def checked(cls, *args, **kwargs):
        kwargs["strict"] = True
        diagnostics = io.StringIO()
        try:
            with contextlib.redirect_stdout(diagnostics):
                policy = original(cls, *args, **kwargs)
        finally:
            print(diagnostics.getvalue(), end="", flush=True)
        if "All keys loaded successfully!" not in diagnostics.getvalue():
            raise RuntimeError(
                "PI05 pretrained weights were not completely loaded. "
                "Stopping instead of training/evaluating a randomly initialized model."
            )
        return policy

    checked._omy_strict = True
    PI05Policy.from_pretrained = classmethod(checked)


def main():
    install_strict_loader()
    from lerobot.scripts.lerobot_train import main as official_main

    official_main()


if __name__ == "__main__":
    main()
