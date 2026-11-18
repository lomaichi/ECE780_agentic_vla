"""Explicit simulator adapter. Callbacks must be validated for the installed LIBERO version."""
import math
from .detectors import Observation

class LiberoAdapter:
    def __init__(self, env, target, destination=None, on_support=None, released=None):
        self.env, self.target, self.destination = env, target, destination
        self.on_support, self.released = on_support, released

    def observe(self, time_s):
        inner = self.env.env
        body_id = inner.obj_body_id[self.target]
        obj = inner.objects_dict[self.target]
        grasp = bool(inner._check_grasp(gripper=inner.robots[0].gripper, object_geoms=obj))
        # robosuite/MuJoCo data API; integration must be checked in the lab.
        body_name = inner.sim.model.body_id2name(body_id)
        linear = inner.sim.data.get_body_xvelp(body_name)
        angular = inner.sim.data.get_body_xvelr(body_name)
        norm = lambda v: math.sqrt(sum(float(x)**2 for x in v))
        return Observation(time_s=time_s,target=self.target,
            target_z_m=float(inner.sim.data.body_xpos[body_id][2]),target_grasped=grasp,
            destination=self.destination,
            on_support=self.on_support(inner,self.target,self.destination) if self.on_support else None,
            released=self.released(inner,self.target) if self.released else None,
            linear_speed_m_s=norm(linear),angular_speed_rad_s=norm(angular))
