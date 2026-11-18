"""fa_core: shared code for the failure-agent pipeline (Phase 1).

  common      constants, paths, pi0 input building, env construction (same settings as record_suites.py)
  predicates  subtask success checks (pick, place, open/close, turn on/off, whole task) over sim state
  motion      scripted end-effector motions (MoveTo, Hold, ActionSequence, Sequence)
  loop        EpisodeRunner: pi0 <-> LIBERO control loop with an agent hook, override and re-prompt,
              trace logging in the record_suites.py format (so any step can be restored later)
  agents      NoOpAgent, ReplayAgent and the agent registry used by the run scripts

Scripts import it by putting failure_agent_prototype/ on sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from fa_core import common, predicates, loop
Every module is Python 3.8 compatible (the LIBERO client venv) and imports LIBERO, robosuite and
openpi_client lazily, so the pure logic can be imported and tested without them.
"""
