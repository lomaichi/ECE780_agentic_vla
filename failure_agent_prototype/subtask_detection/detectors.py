"""Dependency-free, simulator-evidence subtask detectors (Python 3.8+)."""
from dataclasses import dataclass, asdict
from typing import Optional
import math

@dataclass(frozen=True)
class Observation:
    time_s: float
    target: str
    target_z_m: Optional[float] = None
    target_grasped: Optional[bool] = None
    destination: Optional[str] = None
    on_support: Optional[bool] = None
    released: Optional[bool] = None
    linear_speed_m_s: Optional[float] = None
    angular_speed_rad_s: Optional[float] = None
    irrecoverable: Optional[bool] = None

@dataclass(frozen=True)
class Result:
    subtask: str
    status: str
    reason: str
    evidence: dict

    def to_dict(self):
        return asdict(self)

class Detector:
    """One instance per attempt. Terminal results latch; construct anew to retry."""
    def __init__(self, target, start_time_s, hold_s=0.5, timeout_s=10.0, max_gap_s=0.15):
        if not target or not all(math.isfinite(x) for x in (start_time_s,hold_s,timeout_s,max_gap_s)):
            raise ValueError('Target and finite times required')
        if hold_s <= 0 or timeout_s <= hold_s or max_gap_s <= 0:
            raise ValueError('Invalid durations')
        self.target, self.start = target, start_time_s
        self.hold, self.timeout, self.max_gap = hold_s, timeout_s, max_gap_s
        self.last = None
        self.since = None
        self.terminal = None

    def update(self, obs):
        if self.terminal is not None:
            return self.terminal
        if obs.target != self.target:
            raise ValueError('Target mismatch')
        t = obs.time_s
        if not math.isfinite(t) or t < self.start or (self.last is not None and t <= self.last):
            raise ValueError('Timestamps must be finite and strictly increasing')
        if self.last is not None and t-self.last > self.max_gap:
            self.since = None
        self.last = t
        evidence = {'target':self.target, 'elapsed_s':t-self.start}
        if obs.irrecoverable is True:
            status, reason = 'failed', 'irrecoverable_state'
        elif t-self.start >= self.timeout:
            status, reason = 'failed', 'attempt_timeout'
        else:
            valid, reason, detail = self.condition(obs)
            evidence.update(detail)
            if valid is None:
                self.since = None
                status = 'unknown'
            elif not valid:
                self.since = None
                status = 'in_progress'
            else:
                if self.since is None:
                    self.since = t
                duration = t-self.since
                evidence['valid_duration_s'] = duration
                status = 'succeeded' if duration+1e-9 >= self.hold else 'in_progress'
                reason = 'condition_held' if status == 'succeeded' else 'confirming_condition'
        result = Result(self.subtask,status,reason,evidence)
        if status in ('succeeded','failed'):
            self.terminal = result
        return result

class PickDetector(Detector):
    subtask = 'pick'
    def __init__(self, target, start_time_s, initial_height_m, lift_m=0.05, **kwargs):
        super().__init__(target,start_time_s,**kwargs)
        if not math.isfinite(initial_height_m) or not math.isfinite(lift_m) or lift_m <= 0:
            raise ValueError('Invalid lift parameters')
        self.initial_height, self.lift = initial_height_m, lift_m
        self.seen_grasp = False

    def condition(self, obs):
        if obs.target_z_m is None or not math.isfinite(obs.target_z_m) or type(obs.target_grasped) is not bool:
            return None, 'missing_pick_evidence', {}
        lift = obs.target_z_m-self.initial_height
        lost = self.seen_grasp and not obs.target_grasped
        self.seen_grasp |= obs.target_grasped
        return (obs.target_grasped and lift >= self.lift,
                'grasp_lost' if lost else 'pick_not_complete', {'lift_m':lift,'target_grasped':obs.target_grasped})

class PlaceDetector(Detector):
    subtask = 'place'
    def __init__(self, target, destination, start_time_s, max_linear_speed=0.01, max_angular_speed=0.1, **kwargs):
        super().__init__(target,start_time_s,**kwargs)
        if not destination or not all(math.isfinite(x) and x >= 0 for x in (max_linear_speed,max_angular_speed)):
            raise ValueError('Destination and nonnegative finite speed limits required')
        self.destination = destination
        self.max_linear, self.max_angular = max_linear_speed,max_angular_speed

    def condition(self, obs):
        if obs.destination != self.destination:
            raise ValueError('Destination mismatch')
        speeds = (obs.linear_speed_m_s,obs.angular_speed_rad_s)
        if any(type(x) is not bool for x in (obs.on_support,obs.released,obs.target_grasped)) or any(x is None or not math.isfinite(x) or x < 0 for x in speeds):
            return None, 'missing_place_evidence', {}
        stable = speeds[0] <= self.max_linear and speeds[1] <= self.max_angular
        valid = obs.on_support and obs.released and not obs.target_grasped and stable
        return valid, 'place_not_complete', {'destination':self.destination,'on_support':obs.on_support,
                'released':obs.released,'target_grasped':obs.target_grasped,'stable':stable,
                'linear_speed_m_s':speeds[0],'angular_speed_rad_s':speeds[1]}
