"""Single source of truth for the polyhedra classes.

Imported by the Isaac Replicator writer (to map semantic labels -> YOLO class id)
and by the training/inference scripts. Keep this in sync with
``configs/polyhedra.yaml``.

STL -> class mapping (face count in parentheses):
    6C1.STL          -> cube         (6 faces)
    8C1.STL          -> octahedron   (8 faces)
    12C1_Fixed.STL   -> dodecahedron (12 faces)
    20C1.STL         -> icosahedron  (20 faces)
"""

# Ordered by face count. Index == YOLO class id.
CLASS_NAMES = ["cube", "octahedron", "dodecahedron", "icosahedron"]

NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Source STL file (in datasets/) -> class name.
STL_TO_CLASS = {
    "6C1": "cube",
    "8C1": "octahedron",
    "12C1_Fixed": "dodecahedron",
    "20C1": "icosahedron",
}
