"""Agents for fa_core.loop.run_episode, and the registry the run scripts use (--agent NAME).

Phase 1 only needs two:
  noop     never intervenes. Running it must reproduce plain pi0's success rate (PHASE1 step 6);
           if it does not, the loop itself changes pi0's behaviour.
  replay   overrides pi0 with the actions of a recorded episode; used by tools/check_loop.py replay
           to check that the loop reproduces a recording step for step.
Phase 2 adds the real failure-recovery agent here (subclass Agent, add it to AGENTS).
"""

from .loop import CONTINUE, Override, Stop
from .motion import ActionSequence


class Agent(object):
    name = "agent"
    privileged = False     # True only for oracle experiments: also receives a PrivilegedView
    every_steps = 20       # control steps between calls (20 = 1 fps)

    def reset(self, task_language, prompt):
        pass

    def act(self, view, priv=None):
        return CONTINUE


class NoOpAgent(Agent):
    name = "noop"


class ReplayAgent(Agent):
    """Replays recorded actions from the step it is first called at, then stops the episode."""

    name = "replay"

    def __init__(self, actions):
        self.actions = list(actions)
        self.started = False

    def reset(self, task_language, prompt):
        self.started = False

    def act(self, view, priv=None):
        if self.started:
            return Stop("replay finished")
        self.started = True
        return Override(ActionSequence(self.actions[view.step:]), reason="replay")


AGENTS = {"noop": NoOpAgent}


def make_agent(name):
    """Agent by registry name; 'none' means no agent at all (plain pi0 loop)."""
    if name in (None, "none"):
        return None
    if name not in AGENTS:
        raise KeyError(f"unknown agent {name!r}; available: none, {', '.join(sorted(AGENTS))}")
    return AGENTS[name]()
