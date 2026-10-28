"""Build skill_library/library.json (v0.2) from the full survey, in the PoC library's schema.

Entry semantics (detect / recovery / prevention / rl_spec) are written here by hand; `instances`
and RL reset states are filled from survey_results.json so they stay in sync with the survey.
    python aggregate.py && python build_library.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POC = ROOT.parents[1] / "Documentation" / "failure_survey_test" / "skill_library" / "library.json"
results = json.load(open(ROOT / "survey_results.json"))
split = json.load(open(ROOT / "libero90_split.json"))
poc = json.load(open(POC))

primitives = dict(poc["primitives"])
primitives["POINT_APPROACH"] = {
    "args": ["obj_or_point", "dz_m=0.07"],
    "effect": "VLM points at obj in the 1 fps agentview frame; back-project the pixel onto the table/fixture plane "
              "with the known camera calibration; scripted OSC move to dz above it; then hand control back to pi0. "
              "Images + calibration only, no privileged sim positions. New in v0.2: the v1 agent showed 'coordinator "
              "says WHERE, pi0 grasps' is the recipe that works, but used privileged positions (upper bound)"}
primitives["RELEASE"] = {"args": [], "effect": "scripted: lower 2 cm, open gripper, RETREAT up 10 cm. Fallback when pi0 will not let go"}

RETRY = "retry once, then ESCALATE"

ENTRIES = [
    {"id": "HL-1", "level": "high", "name": "subtask_handoff_stall",
     "summary": "pi0 completes a subtask (a fixture action or a successful grasp) and then idles; the next subtask never starts.",
     "applies_when": "compound instruction, or a grasp that should be followed by transport; subtask k verified done; next target visible and not being approached",
     "slots": ["done_subtask", "next_obj", "next_dest"],
     "detect": {"proprio": "EE displacement < 1 cm across 3 frames, OR EE out of camera view for 2 frames",
                "visual": "VERIFY(done_subtask complete) == true AND EE not moving toward next_obj/next_dest",
                "earliest_fire_s": 3,
                "distinguish_from": "LL-7/LL-8 hesitation: there the EE is next to the *current* target making small adjustments; LL-1: current subtask not complete"},
     "recovery": [
         {"do": "VERIFY", "args": {"predicate": "{done_subtask} is complete"}, "on_false": "not HL-1: re-classify"},
         {"do": "HALT"},
         {"do": "RETREAT", "args": {"dz_m": 0.10}, "why": "leave the finished fixture; also brings an out-of-view arm back into view"},
         {"do": "REPROMPT", "args": {"instruction": "next-subtask sentence, e.g. pick_place_on(next_obj, next_dest); if an object is already held, the place sentence for it"}},
         {"do": "WATCH", "args": {"cue": "EE moves toward next_obj/next_dest", "window_s": 3}, "on_not_fired": "POINT_APPROACH(next_obj) + REPROMPT, " + RETRY}],
     "prevention": "Coordinator issues one subtask at a time and advances only on VERIFY.",
     "success_check": "VERIFY(next subtask started: EE at next_obj / object moving to next_dest) within 10 s"},

    {"id": "HL-2", "level": "high", "name": "scene_default_skill_override",
     "summary": "pi0 runs the scene's habitual skill or fixture interaction (usually from a trained libero_10/libero_spatial task) instead of the instructed skill, e.g. pushes the open drawer shut, grabs a moka pot when told to turn the knob, skips the drawer and puts the bowl on the plate.",
     "applies_when": "current subtask is skill S on target X, but the EE heads for a different fixture/object Y that belongs to a trained task in this scene",
     "slots": ["target", "wrong_fixture", "destination", "protected_state"],
     "detect": {"proprio": "EE path over 2 frames brings it closer to wrong_fixture and away from target, gripper empty",
                "visual": "VLM: 'is the gripper approaching {target}?' == no AND 'is it at/pushing {wrong_fixture}?' == yes",
                "protected_state_check": "VERIFY({protected_state}) every frame, e.g. 'top drawer is open'",
                "earliest_fire_s": 2,
                "distinguish_from": "HL-3: there the *skill* is right (pick-place) but the object is wrong"},
     "recovery": [
         {"do": "HALT"},
         {"do": "RETREAT", "args": {"dz_m": 0.10, "dx_m": "-0.05 (away from wrong_fixture)"}, "why": "remove the attractor; pi0's skill choice is driven by what is near the gripper"},
         {"do": "VERIFY", "args": {"predicate": "{protected_state}"}, "on_false": {"do": "INSERT_SUBTASK", "args": {"subtask": "restore protected state, e.g. open_fixture(destination)", "position": "next"}}},
         {"do": "POINT_APPROACH", "args": {"obj_or_point": "{target}"}},
         {"do": "REPROMPT", "args": {"instruction": "object-first sentence for the instructed subtask"}},
         {"do": "WATCH", "args": {"cue": "EE moves toward wrong_fixture again", "window_s": 5}, "on_fired": "repeat with RETREAT dz 0.15, then ESCALATE"}],
     "prevention": "Declare protected states before running pi0. For fixture tasks (knob, drawer handle) POINT_APPROACH the fixture before the first pi0 step: the trained-task attractor wins within the first second.",
     "success_check": "VERIFY(gripper at {target}) within 5 s AND protected_state still true"},

    {"id": "HL-3", "level": "high", "name": "wrong_object_category",
     "summary": "pi0 grasps/moves an object of a different category than the target, typically one from a trained task in the same scene (alphabet soup instead of tomato sauce, white bowl instead of frying pan, mug instead of book).",
     "applies_when": "subtask pick(target); gripper descending on / closing on an object",
     "slots": ["target", "wrong_obj", "destination"],
     "detect": {"proprio": "pre-grasp trigger: EE z < ~5 cm above the support surface OR gripper width decreasing",
                "visual": "VERIFY('the object between the fingers is the {target}', 2x crop around the gripper)",
                "earliest_fire_s": 2,
                "distinguish_from": "HL-4: same category, wrong instance (qualifier ignored)"},
     "recovery": [
         {"do": "HALT"},
         {"do": "if_lifted", "steps": [{"do": "RELEASE", "why": "set wrong_obj back down before moving on"}]},
         {"do": "RETREAT", "args": {"dz_m": 0.10}},
         {"do": "POINT_APPROACH", "args": {"obj_or_point": "{target}"}, "why": "re-prompting alone loses to the scene prior (the original prompt already did)"},
         {"do": "REPROMPT", "args": {"instruction": "pick_place_{rel}(target, destination)"}},
         {"do": "VERIFY", "args": {"predicate": "{target} is in the gripper"}, "on_false": RETRY}],
     "prevention": "Coordinator lists the scene's trained-task objects as known distractors; FR agent runs the pre-grasp VERIFY on every pick. In scenes that share objects with a libero_10 task, start with POINT_APPROACH rather than a plain prompt.",
     "success_check": "VERIFY({target} grasped and lifted) within 10 s"},

    {"id": "HL-4", "level": "high", "name": "referring_expression_instance_error",
     "summary": "Right kind of object or fixture, wrong instance: the qualifier (left/middle/right, colour, top/bottom drawer) is ignored. pi0 opens the middle drawer for 'top'/'bottom', always grabs the same book, takes the white mug for 'red mug'.",
     "applies_when": "target is named with a qualifier and the scene has >1 instance of its category",
     "slots": ["target", "qualifier", "wrong_instance"],
     "detect": {"proprio": "same pre-grasp / pre-pull trigger as HL-3",
                "visual": "VERIFY phrased on the qualifier on a 3x zoom: 'is the gripper on the lowest handle?', 'is this the red mug?', 'is this the middle book?'",
                "earliest_fire_s": 2,
                "distinguish_from": "HL-3: different category"},
     "recovery": [
         {"do": "HALT"},
         {"do": "RETREAT", "args": {"dz_m": 0.10}},
         {"do": "optional", "steps": [{"do": "INSERT_SUBTASK", "args": {"subtask": "undo wrong_instance change, e.g. close the wrongly opened drawer"}}]},
         {"do": "POINT_APPROACH", "args": {"obj_or_point": "{target} as resolved by the coordinator"}},
         {"do": "REPROMPT", "args": {"instruction": "original sentence for the subtask"}},
         {"do": "VERIFY", "args": {"predicate": "qualifier-specific check"}, "on_false": RETRY}],
     "prevention": "The coordinator resolves the qualifier itself, in the robot frame (robot-left = IMAGE RIGHT; front = lower in the image; verified on libero_10 task4), and gives pi0 a pointed target, never the qualifier alone.",
     "success_check": "VERIFY(correct instance grasped / correct drawer moving)"},

    {"id": "HL-5", "level": "high", "name": "intermediate_subtask_skipped",
     "summary": "In 'stack A on B and place them in C', pi0 skips the stack and carries a single bowl straight to C. It never stacked in any surveyed episode.",
     "applies_when": "instruction with an intermediate stack/combine subtask before the final place",
     "slots": ["top_obj", "base_obj", "container"],
     "detect": {"proprio": "loaded gripper moving toward container before VERIFY(stacked)",
                "visual": "VLM: 'is {top_obj} resting inside {base_obj}?' == no while the EE heads to {container}",
                "earliest_fire_s": 2},
     "recovery": [
         {"do": "HALT"},
         {"do": "branch", "if": "base_obj already in container", "then": [
             {"do": "REPROMPT", "args": {"instruction": "pick up the {top_obj} and place it on the {base_obj}"}, "why": "stacking in place satisfies the goal"}],
          "else": [
             {"do": "RELEASE"}, {"do": "POINT_APPROACH", "args": {"obj_or_point": "{top_obj}"}},
             {"do": "REPROMPT", "args": {"instruction": "pick up the {top_obj} and place it on the {base_obj}"}},
             {"do": "VERIFY", "args": {"predicate": "{top_obj} rests in {base_obj}"}},
             {"do": "REPROMPT", "args": {"instruction": "pick up the {base_obj} and place it in the {container}"}}]},
         {"do": "WATCH", "args": {"cue": "stack in container", "window_s": 5}, "on_not_fired": RETRY}],
     "prevention": "Split 'stack and place' into two VERIFY-gated prompts.",
     "success_check": "VERIFY(top_obj in base_obj AND base_obj in container)"},

    {"id": "HL-6", "level": "high", "name": "destination_misgrounding",
     "summary": "pi0 holds the correct object but carries it to / hovers at / places it in a non-instructed destination or relation, usually the trained task's destination (back compartment for 'front', right of the plate for 'left of', microwave for 'in front of the mug'), or hovers without committing.",
     "applies_when": "object grasped; subtask place(obj, dest, relation)",
     "slots": ["obj", "destination", "relation", "wrong_destination"],
     "detect": {"proprio": "loaded gripper moving away from the destination over 2 frames, OR hovering > 3 s without nearing it",
                "visual": "VLM: 'is the held {obj} moving toward / above {destination} ({relation})?' == no",
                "earliest_fire_s": 2,
                "distinguish_from": "LL-7: the object is AT the correct destination but not released"},
     "recovery": [
         {"do": "HALT", "why": "keep the grasp"},
         {"do": "RETREAT", "args": {"dz_m": 0.10}},
         {"do": "POINT_APPROACH", "args": {"obj_or_point": "point resolved from {destination} + {relation}"}},
         {"do": "REPROMPT", "args": {"instruction": "pick_place_{rel}(obj, destination)"}, "fallback": "RELEASE if pi0 still will not lower"},
         {"do": "VERIFY", "args": {"predicate": "{obj} {relation} {destination}"}, "on_false": RETRY}],
     "prevention": "The coordinator turns relation/compartment words into a target point before the carry; pi0 is not trusted with relations. 'Under the shelf' may need a horizontal-insertion skill (open gap from v1).",
     "success_check": "VERIFY({obj} {relation} {destination})"},

    {"id": "HL-7", "level": "high", "name": "target_never_approached",
     "summary": "pi0 hovers/wanders among other objects and fixtures for the whole episode and never approaches the instructed target (e.g. a bowl in an unusual raised start pose on top of the microwave).",
     "applies_when": "any subtask; no progress toward target after the first seconds",
     "slots": ["target"],
     "detect": {"proprio": "EE-to-target distance (from VLM pointing) not decreasing over 3 frames",
                "visual": "VLM heading check: 'is the gripper moving toward {target}?' == no for 3 frames",
                "earliest_fire_s": 3,
                "distinguish_from": "HL-2/HL-3: there pi0 commits to a specific wrong fixture/object"},
     "recovery": [
         {"do": "HALT"}, {"do": "RETREAT", "args": {"dz_m": 0.10}},
         {"do": "POINT_APPROACH", "args": {"obj_or_point": "{target}"}},
         {"do": "REPROMPT", "args": {"instruction": "original sentence"}},
         {"do": "WATCH", "args": {"cue": "EE at target", "window_s": 3}, "on_not_fired": "ESCALATE"}],
     "prevention": "Heading check in the first 2-3 frames instead of waiting for the timeout.",
     "success_check": "VERIFY(EE at {target}) within 5 s"},

    {"id": "HL-8", "level": "high", "name": "grasp_loss_unnoticed",
     "summary": "After a missed or slipped grasp (LL-4/LL-5), pi0 continues the transport/place motion with an empty gripper instead of re-grasping.",
     "applies_when": "after any pick; object no longer between the fingers",
     "slots": ["obj", "destination"],
     "detect": {"proprio": "gripper width ~0 (fully closed) while EE moves toward destination",
                "visual": "VERIFY('{obj} is between the fingers', crop) == false",
                "earliest_fire_s": 1,
                "distinguish_from": "a deliberate release at the destination"},
     "recovery": [
         {"do": "HALT"}, {"do": "RETREAT", "args": {"dz_m": 0.10}},
         {"do": "POINT_APPROACH", "args": {"obj_or_point": "{obj} at its new pose"}},
         {"do": "INVOKE_RL", "args": {"policy_id": "rl/pick"}, "available": False, "fallback": "REPROMPT pick_place sentence"},
         {"do": "VERIFY", "args": {"predicate": "{obj} grasped"}, "on_false": RETRY}],
     "prevention": "VERIFY the grasp after every pick before allowing transport; this turns every LL-4/LL-5 into a recoverable event.",
     "success_check": "VERIFY({obj} grasped and lifted)"},
]

RL = {
    "rl/close_drawer": {"subtask": "close(drawer)",
        "env_and_reward": "libero_90 task22 'close the bottom drawer of the cabinet' (survey split): its LIBERO success predicate is exactly this subtask. NOTE: the PoC used libero_90 task0, which is now in the HELD-OUT test half, so task0/task2 states are no longer used",
        "guide_demos": [{"episode": "libero_90/task22_success_init0", "why": "same task solved by pi0 (not yet watched)"},
                        {"episode": "libero_10/task3_success_init0", "why": "PoC-verified: horizontal push with the hand beside the drawer front, steps ~210-238"},
                        {"episode": "libero_10/task3_success_init1", "why": "same task, second init"}],
        "horizon_steps": 120},
    "rl/open_door": {"subtask": "open(hinged door)",
        "env_and_reward": "libero_90 task35 'open the microwave' (survey split), LIBERO predicate",
        "guide_demos": [{"episode": "none in-distribution", "why": "libero_10 task9 only CLOSES the microwave; consider a scripted pull demo"}],
        "horizon_steps": 200},
    "rl/turn_knob": {"subtask": "turn_on(stove) / turn_off(stove)",
        "env_and_reward": "libero_90 task44 'turn on the stove' (survey split), LIBERO predicate; task39 'turn off the stove' for the reverse",
        "guide_demos": [{"episode": "libero_10/task2_success_init0", "why": "KITCHEN_SCENE3: pi0 turns the stove on reliably (not yet watched)"}],
        "horizon_steps": 120},
    "rl/pick": {"subtask": "pick(obj): stable grasp and lift, no disturbance, within a time budget",
        "env_and_reward": "custom per-subtask predicate (Grasped(obj) AND obj lifted >= 5 cm for 10 steps) on the reset scene; whole-task LIBERO success is too sparse. Add a step penalty / short horizon for LL-8",
        "guide_demos": [{"episode": "libero_spatial/*_success_*", "why": "bowl picks (not yet watched)"},
                        {"episode": "libero_10/task5_success_init0", "why": "book pick (not yet watched)"}],
        "horizon_steps": 100},
    "rl/place": {"subtask": "place(obj, dest, relation) incl. release",
        "env_and_reward": "custom predicate On/In(obj, dest) AND gripper open AND EE retreated, from a reset with obj held above dest",
        "guide_demos": [{"episode": "libero_spatial/*_success_*", "why": "bowl-on-plate places (not yet watched)"},
                        {"episode": "libero_10/task4_success_init0", "why": "mug-on-plate places (left/right verified)"}],
        "horizon_steps": 100},
}
RL_STATE_SOURCE = "trace/states.npy[step] of that episode; follow state_convention in meta.json (controller update, gripper action, qacc_warmstart)"

LL = [
    ("LL-1", "drawer_push_stall", "rl/close_drawer",
     "While closing a drawer pi0 pushes from above; the drawer travels 1/2-2/3 of the way, then the hand rests on the drawer front.",
     "subtask close(drawer); drawer not flush; EE at drawer front",
     {"proprio": "EE displacement along the closing direction < 1 cm across 2 frames while at the drawer",
      "visual": "VERIFY('{drawer} front is flush with the cabinet', 2x zoom) == false",
      "distinguish_from": "HL-1: drawer already flush"},
     ["drawer"]),
    ("LL-2", "hinged_door_pull_stall", "rl/open_door",
     "pi0 hooks a hinged door's handle and swings it partly open, then stops pulling short of the open threshold.",
     "subtask open(door); door partly open; EE at handle",
     {"proprio": "EE stationary 2 frames while at the handle", "visual": "VERIFY('{door} is wide open', zoom) == false",
      "distinguish_from": "HL-7: door never touched"},
     ["door"]),
    ("LL-3", "knob_approach_contortion", "rl/turn_knob",
     "Approaching a stove knob the wrist/elbow folds into a twisted configuration, the wrist body touches the knob instead of the fingertips, and the arm swings away (often out of view) without turning it.",
     "subtask turn_on/turn_off(stove); EE near the knob",
     {"proprio": "large wrist-joint rotation (> 90 deg) within 1 s near the knob, or EE leaves the view",
      "visual": "VLM: 'are the fingertips on the knob?' == no while the wrist touches it",
      "distinguish_from": "HL-2 in KITCHEN_SCENE8: there the EE never goes to the knob"},
     ["knob"]),
    ("LL-4", "unstable_grasp_slip", "rl/pick",
     "The grasp closes on a rim, edge or handle; during lift or transport the object tilts, pivots or slips out of the gripper (bowl rims, book top edges, mug rims, pan handles).",
     "after a grasp on obj",
     {"proprio": "gripper width drops to ~0 during lift/transport (object lost), or EE rises but obj stays",
      "visual": "VLM on crop: 'is {obj} tilted / hanging / back on the table?'",
      "distinguish_from": "LL-5: contact disturbance BEFORE a grasp forms"},
     ["obj"]),
    ("LL-5", "pregrasp_contact_disturbance", "rl/pick",
     "A misaligned descent knocks over, pushes or tips the target (or a neighbouring receptacle) before a grasp is formed.",
     "subtask pick(obj); EE descending",
     {"proprio": "contact during descent with gripper still open", "visual": "VLM: 'did {obj} fall over / move?' comparing consecutive frames",
      "distinguish_from": "LL-4: grasp formed first"},
     ["obj"]),
    ("LL-6", "placement_instability", "rl/place",
     "The object is released off-centre, at an edge or tilted, so it falls off the support, tips over or overhangs the region.",
     "subtask place(obj, dest); right after release",
     {"proprio": "gripper opened at destination", "visual": "VERIFY('{obj} sits upright and fully {relation} {dest}', zoom) == false",
      "distinguish_from": "HL-6: wrong destination"},
     ["obj", "dest"]),
    ("LL-7", "place_release_stall", "rl/place",
     "The object is brought to (or just beside) the correct destination but pi0 keeps adjusting and never lowers/releases it before timeout.",
     "subtask place(obj, dest); obj held at dest",
     {"proprio": "gripper closed, EE within ~5 cm of dest, displacement < 1 cm over 3 frames",
      "visual": "VLM: 'is {obj} held above/next to {dest}?' == yes",
      "distinguish_from": "HL-6: at a wrong destination"},
     ["obj", "dest"]),
    ("LL-8", "pregrasp_hesitation", "rl/pick",
     "The gripper hovers next to the correct target with small adjustments for many seconds before (or instead of) closing, exhausting the step budget.",
     "subtask pick(obj); EE next to the correct obj, gripper open",
     {"proprio": "gripper open, EE within ~5 cm of obj, net displacement < 1 cm over 3 frames",
      "visual": "VLM: 'is the gripper around {obj} but not closed?'",
      "distinguish_from": "HL-7: EE not near the target at all"},
     ["obj"]),
]

# instances from the survey
inst = {}
for e in results["episodes"]:
    for f in e["failures"]:
        i = {"episode": e["episode"], "onset_step": f["onset_step"], "onset_window": f["onset_window"],
             "step_resolution": e["step_resolution"], "primary": f["primary"], "subtask": f["subtask"],
             "confidence": f["confidence"]}
        if f["level"] == "high":
            i["episode_remedy"] = f.get("hl_remedy")
        else:
            i["rl_reset_step"] = f.get("rl_reset_step")
            i["on_instructed_subtask"] = "[no" not in f["subtask"]
        inst.setdefault(f["mode_id"], []).append(i)

entries = []
for e in ENTRIES:
    entries.append({**e, "instances": inst.get(e["id"], []), "status": "proposed"})
for mid, name, pol, summary, when, detect, slots in LL:
    resets = [{"episode": i["episode"], "step": i["rl_reset_step"], "onset_step": i["onset_step"]}
              for i in inst.get(mid, []) if i["on_instructed_subtask"] and i["rl_reset_step"] is not None]
    entries.append({
        "id": mid, "level": "low", "name": name, "summary": summary, "applies_when": when, "slots": slots,
        "detect": {**detect, "earliest_fire_s": 2},
        "recovery": [
            {"do": "INVOKE_RL", "args": {"policy_id": pol}, "available": False},
            {"do": "fallback_until_rl", "steps": [
                {"do": "HALT"}, {"do": "RETREAT", "args": {"dz_m": 0.05}},
                {"do": "POINT_APPROACH", "args": {"obj_or_point": "{%s}" % slots[0]}},
                {"do": "REPROMPT", "args": {"instruction": "original subtask sentence"}},
                {"do": "WATCH", "args": {"cue": "subtask success_check", "window_s": 5}, "on_not_fired": "ESCALATE"}],
             "status": "proposed, expected weak (re-runs the same pi0 behaviour)"}],
        "rl_spec": {"policy_id": pol, **{k: v for k, v in RL[pol].items()}, "method": "SimpleVLA-RL on pi0",
                    "reset_states": resets, "reset_state_source": RL_STATE_SOURCE},
        "instances": inst.get(mid, []),
        "status": "proposed (RL not started)"})

rl_policies = {}
for pol, spec in RL.items():
    modes = [m for m, _, p, *_ in LL if p == pol]
    rl_policies[pol] = {**spec, "covers": modes, "method": "SimpleVLA-RL on pi0",
                        "reset_states": [r for en in entries if en["id"] in modes for r in en["rl_spec"]["reset_states"]],
                        "reset_state_source": RL_STATE_SOURCE}

out = {
    "name": "pi0 failure recovery skill library",
    "version": "0.2",
    "built_from": "failure_agent_prototype/failure_survey (87 failures: libero_10 x5, libero_90 x82 from the 45-task survey split; video only)",
    "supersedes": "0.1-poc (Documentation/failure_survey_test/skill_library); HL-1, HL-2, LL-1 keep their IDs and meaning",
    "split_guard": {"note": "only survey-split libero_90 tasks may be used for RL envs, reset states or demos",
                    "held_out_tasks": split["held_out_tasks"]},
    "status_legend": poc["status_legend"],
    "observation_budget": poc["observation_budget"],
    "camera_convention": "agentview faces the robot: robot-left = IMAGE RIGHT, front = lower in the image (verified on libero_10/task4_success_init0)",
    "primitives": primitives,
    "prompt_templates": poc["prompt_templates"],
    "rl_policies": rl_policies,
    "entries": entries,
}
(ROOT / "skill_library").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "skill_library" / "library.json", "w"), indent=2)
for en in entries:
    print(en["id"], en["name"], len(en["instances"]), "resets" if en["level"] == "low" else "",
          len(en["rl_spec"]["reset_states"]) if en["level"] == "low" else "")
