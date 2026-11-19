"""Paired different-rule Q11 entry point; reuses the same exact native backend."""
from .vqa import run as run_q11

def run(config,args):
    args.task='rule'
    return run_q11({**config,'task':'rule'},args)
