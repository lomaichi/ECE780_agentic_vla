"""BDDL names -> natural-language phrases, and prompt slot filling.

pi0_libero was trained on LIBERO instructions, so recovery prompts reuse the
same phrasing style ("pick up the X and place it in the Y", "close the top
drawer of the cabinet").
"""
from __future__ import annotations

import re
import string
from typing import Dict, List, Optional

import numpy as np

# BDDL object category -> phrase (category = name without the trailing _<n>)
NAME_MAP = {
    "akita_black_bowl": "black bowl",
    "new_salad_dressing": "salad dressing",
    "salad_dressing": "salad dressing",
    "chefmate_8_frypan": "frying pan",
    "flat_stove": "stove",
    "moka_pot": "moka pot",
    "wooden_cabinet": "cabinet",
    "white_cabinet": "cabinet",
    "wooden_tray": "tray",
    "basket": "basket",
    "black_book": "book",
    "yellow_book": "book",
    "desk_caddy": "caddy",
    "alphabet_soup": "alphabet soup",
    "tomato_sauce": "tomato sauce",
    "orange_juice": "orange juice",
    "cream_cheese": "cream cheese",
    "chocolate_pudding": "chocolate pudding",
    "red_coffee_mug": "red mug",
    "white_yellow_mug": "yellow and white mug",
    "porcelain_mug": "white mug",
    "plate": "plate",
    "microwave": "microwave",
    "ketchup": "ketchup",
    "butter": "butter",
    "milk": "milk",
    "wine_bottle": "wine bottle",
    "wine_rack": "wine rack",
    "white_bowl": "white bowl",
    "wooden_two_layer_shelf": "cabinet shelf",
}

# Region suffix -> noun phrase including its article ("on top of the cabinet")
REGION_MAP = {
    "top_region": "the top drawer of the {owner}",
    "middle_region": "the middle drawer of the {owner}",
    "bottom_region": "the bottom drawer of the {owner}",
    "top_side": "top of the {owner}",
    "cook_region": "the {owner}",
    "contain_region": "the {owner}",
    "left_contain_region": "the left compartment of the {owner}",
    "right_contain_region": "the right compartment of the {owner}",
    "front_contain_region": "the front compartment of the {owner}",
    "back_contain_region": "the back compartment of the {owner}",
    "heating_region": "the {owner}",
}

# (owner category, region suffix) overrides where the generic phrase is wrong
REGION_OVERRIDES = {
    ("wooden_two_layer_shelf", "top_region"): "the cabinet shelf",
    ("wooden_two_layer_shelf", "bottom_region"): "the cabinet shelf",
    ("wooden_two_layer_shelf", "top_side"): "top of the cabinet",
    ("wine_rack", "top_region"): "the wine rack",
}

# Relation word for a placement, when it is not the predicate's own ("in"/"on")
REL_OVERRIDES = {
    ("wooden_two_layer_shelf", "top_region"): "on",
    ("wooden_two_layer_shelf", "bottom_region"): "under",
}

# Table regions defined relative to another object:
# kitchen_table_plate_right_region -> "to the right of the plate"
_TABLE_REL = re.compile(r"^(?:kitchen|living_room|study|main|floor)?_?table_(.+)_(left|right|front|back)_region$")

VERB_MAP = {"open": "open", "close": "close", "turnon": "turn on", "turnoff": "turn off"}


def category(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


def obj_phrase(name: str) -> str:
    cat = category(name)
    return NAME_MAP.get(cat, cat.replace("_", " "))


def region_phrase(name: str, owner: str) -> str:
    """'wooden_cabinet_1_top_region' with owner 'wooden_cabinet_1' -> 'the top drawer of the cabinet'."""
    m = _TABLE_REL.match(name)
    if m:
        return f"the {m.group(2)} of the {NAME_MAP.get(m.group(1), m.group(1).replace('_', ' '))}"
    if name == owner:
        return "the " + obj_phrase(owner)
    suffix = name[len(owner) + 1:] if name.startswith(owner + "_") else name
    tmpl = REGION_OVERRIDES.get((category(owner), suffix)) or REGION_MAP.get(suffix, "the {owner}")
    return tmpl.format(owner=obj_phrase(owner))


def place_phrase(rel: str, dest: str, owner: str, pos: Optional[Dict[str, list]] = None) -> str:
    """Destination clause of a placement, LIBERO style: 'in the basket',
    'on top of the cabinet', 'to the right of the plate', 'under the cabinet shelf',
    'on the black bowl on the right' (an object destination with look-alikes)."""
    m = _TABLE_REL.match(dest)
    if m:
        noun = NAME_MAP.get(m.group(1), m.group(1).replace("_", " "))
        return f"to the {m.group(2)} of the {noun}"
    suffix = dest[len(owner) + 1:] if dest.startswith(owner + "_") else ""
    rel = REL_OVERRIDES.get((category(owner), suffix), rel)
    desc = spatial_descriptor(pos, dest) if pos else None
    return f"{rel} {region_phrase(dest, owner)}" + (f" {desc}" if desc else "")


def spatial_descriptor(pos: Dict[str, "list"], name: str) -> Optional[str]:
    """Disambiguate `name` from other objects with the same noun phrase (two
    black bowls, two plates, a black and a yellow book) by world position.
    LIBERO's instructions use the robot-base convention: "left" is world -y,
    "front" is world +x (in the image pi0 sees, left/right are mirrored, since
    the camera faces the robot). Returns e.g. 'on the left', 'in the middle',
    or None if the object is unique."""
    group = {k: v for k, v in pos.items() if obj_phrase(k) == obj_phrase(name)}
    if name not in pos or len(group) < 2:
        return None
    pts = {k: (float(v[0]), float(v[1])) for k, v in group.items()}
    # axis along which the look-alikes are spread most: 0 = x (back/front), 1 = y (left/right)
    ax = int(np.argmax([np.ptp([p[i] for p in pts.values()]) for i in (0, 1)]))
    vals = sorted(p[ax] for p in pts.values())
    v = pts[name][ax]
    if len(vals) >= 3 and vals[0] < v < vals[-1]:
        return "in the middle"
    others = [p[ax] for k, p in pts.items() if k != name]
    lower = v < sum(others) / len(others)
    if ax == 1:
        return "on the left" if lower else "on the right"
    return "at the back" if lower else "at the front"


def fill(template: str, slots: Dict[str, str]) -> Optional[str]:
    """Fill a prompt template; None if a slot it needs is unavailable."""
    needed = [f for _, f, _, _ in string.Formatter().parse(template) if f]
    if any(not slots.get(f) for f in needed):
        return None
    return template.format(**{f: slots[f] for f in needed})


def choose_prompt(variants: List[str], slots: Dict[str, str], attempt: int) -> Optional[str]:
    """Variant `attempt` (wrapping), skipping variants with missing slots."""
    n = len(variants)
    for k in range(n):
        p = fill(variants[(attempt + k) % n], slots)
        if p:
            return p
    return None
