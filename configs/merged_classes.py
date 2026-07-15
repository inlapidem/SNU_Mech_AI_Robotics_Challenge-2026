"""Runtime/training alias for the UNIFIED class space.

The single source of truth is now configs/combined_classes.py (the same file the
unified Isaac generator sim/generate_combined_data.py uses to emit the dataset), so
the classifier's label space is defined in exactly ONE place. This module re-exports
it unchanged so existing runtime/training/deploy imports
(`from configs.merged_classes import set_of`, CLASSIFIER_CLASSES, ...) keep working.

Roles: combined_* = unified DATA/synthesis; merged_* = unified RUNTIME/model/deploy.
Both share this taxonomy. See docs/long_range_upgrade.md "통합 인식 파이프라인".
"""

from configs.combined_classes import (  # noqa: F401
    SHAPE_CLASSES, FRUIT_CLASSES, UNKNOWN,
    SET1_SHAPES, SET2_FRUITS,
    CLASSIFIER_CLASSES, NAME_TO_ID, DETECTOR_CLASSES,
    CLASS_TO_SET, set_of, targets_from_list,
    SHAPE_CONF_GATE, FRUIT_CONF_GATE,
)
