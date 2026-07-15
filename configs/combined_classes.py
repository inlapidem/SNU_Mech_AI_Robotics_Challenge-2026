"""Single source of truth for the UNIFIED (Set 1 + Set 2) perception task.

Rulebook: both object families share the arena and the match runs simultaneously, so
the robot recognizes everything with ONE detector + ONE classifier (half the Jetson
inference of the old per-set split, and a training distribution that matches deployment
exactly). This replaces configs/classes.py (Set 1) and configs/set2_classes.py (Set 2)
for the combined pipeline; those stay for the legacy per-set datasets/models.

Two label spaces:

  * DETECTOR  -> single class 'object'. Localizes EVERY collectible (any polyhedron OR
    any cube) and does NOT identify it. Class id is always 0. There are no "hard
    negative" objects anymore: a polyhedron used to be a Set 2 negative, but here it is
    a positive the robot must also detect. The only negatives are venue clutter
    (wood/stickers/tape) via deliberate object-free frames.

  * CLASSIFIER -> 4 shapes + 4 fruits + 'unknown' (9 classes). Labels come from VISIBLE
    evidence, never hidden identity:
      - octahedron/dodecahedron/icosahedron: unambiguous by shape.
      - 'cube': the WHITE-CUBE appearance. A Set 1 cube and a Set 2 fruit cube whose
        fruit faces are not visible are PHYSICALLY IDENTICAL from a blank view, so they
        share this one label (identical pixels must never carry two labels, or the
        classifier is poisoned). The robot disambiguates by approaching and checking
        for a fruit face (runtime policy), not by guessing from one frame.
      - apple/orange/banana/pineapple: a cube with sufficient visible fruit evidence.
      - 'unknown': reliability reject ONLY (crop too small / occluded / truncated /
        background). No cross-set 'unknown' injection is needed - fruit cubes and
        polyhedra are all real classes now.
"""

SHAPE_CLASSES = ["cube", "octahedron", "dodecahedron", "icosahedron"]
FRUIT_CLASSES = ["apple", "orange", "banana", "pineapple"]
UNKNOWN = "unknown"
# Aliases used by the runtime/training side (configs/merged_classes.py re-exports these).
SET1_SHAPES = SHAPE_CLASSES
SET2_FRUITS = FRUIT_CLASSES

# Classifier output space (ordered for reference; the TRAINED order is whatever
# torchvision ImageFolder assigns alphabetically, persisted to classes.json, so runtime
# maps by NAME not index). Index == classifier class id here.
CLASSIFIER_CLASSES = SHAPE_CLASSES + FRUIT_CLASSES + [UNKNOWN]
NAME_TO_ID = {name: i for i, name in enumerate(CLASSIFIER_CLASSES)}

# Detector output space: one class ('object' = any polyhedron OR cube).
DETECTOR_CLASSES = ["object"]

# Pool sizing for the generator.
CUBES_PER_FRUIT = 3          # fruit cubes per fruit class -> 12 fruit cubes
POLYS_PER_SHAPE = 2          # copies of each polyhedron shape -> 8 polyhedra

# Runtime confidence regime differs by family (fruit mis-pickup = -40, shapes cheaper),
# so the unified classifier keeps a per-family acceptance gate. The runtime side
# (configs/merged.yaml) carries the numeric thresholds; these are the reference values.
SHAPE_CONF_GATE = 0.60
FRUIT_CONF_GATE = 0.90

# class name -> owning set ("set1" | "set2"); 'unknown' owns no set (None). The runtime
# (runtime/merged_pipeline.py) DERIVES a detection's set from its predicted class and
# routes to the existing per-set decision policy, so the mission FSM's per-set target
# model works unchanged.
CLASS_TO_SET = {**{s: "set1" for s in SHAPE_CLASSES},
                **{f: "set2" for f in FRUIT_CLASSES},
                UNKNOWN: None}


def set_of(cls):
    """Owning set of a predicted class name: 'set1', 'set2', or None ('unknown'/misc)."""
    return CLASS_TO_SET.get(cls)


def targets_from_list(names):
    """['cube', 'apple'] -> {'set1': 'cube', 'set2': 'apple'} (one target per set).

    Rejects an unknown object name or two targets that resolve to the same set."""
    targets = {}
    for name in names:
        s = CLASS_TO_SET.get(name)
        if s is None:
            raise ValueError(
                f"--target '{name}' is not a valid object; choose from "
                f"{SHAPE_CLASSES + FRUIT_CLASSES}")
        if s in targets:
            raise ValueError(
                f"--target has two objects in {s} ('{targets[s]}' and '{name}'); "
                f"the mission holds one target per set (e.g. one shape + one fruit)")
        targets[s] = name
    return targets
