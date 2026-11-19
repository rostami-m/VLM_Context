"""Run prompt-only, fixed and triggered policies on the paper's one-order panel."""
from .continual import run_streams


def run(config, args):
    return run_streams(config, args, controller=True)
