"""Single source of truth for the Set 2 fruit-cube classes.

Imported by the Set 2 Isaac generator (to map semantic labels -> ids), the
training/inference scripts, and the runtime decision policy. Kept SEPARATE from
configs/classes.py (Set 1 polyhedra) on purpose: Set 1 and Set 2 are independent
perception tasks (different datasets, models, thresholds, runtime policies).

Two label spaces:

  * DETECTOR  -> single class 'cube_candidate'. The detector localizes cube-like
    objects and does NOT classify the fruit. Class id is always 0.

  * CLASSIFIER -> fruit classes + 'unknown'. The crop is labelled with a fruit
    class ONLY when there is sufficient *visible fruit evidence* on a cube face;
    otherwise 'unknown' (plain white face, fruit too small/occluded/blurred, a
    Set 1 cube, a false-positive crop, or background). The classifier learns
    visible-fruit recognition, never hidden-cube identity guessing.
"""

# Detector: one class only.
DETECTOR_CLASSES = ["cube_candidate"]

# Classifier: 4 fruit classes (ordered) + the reject class.
FRUIT_CLASSES = ["apple", "orange", "banana", "pineapple"]
UNKNOWN = "unknown"
CLASSIFIER_CLASSES = FRUIT_CLASSES + [UNKNOWN]

NAME_TO_ID = {name: i for i, name in enumerate(CLASSIFIER_CLASSES)}

# Each physical Set 2 cube belongs to exactly one fruit class and carries that
# fruit image on 3 of its 6 faces. 3 cubes per class -> 12 cubes total.
CUBES_PER_CLASS = 3
