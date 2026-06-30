---
name: set2-fruit-cube-pipeline
description: "Set 2 perception build — fruit-cube detector+classifier, key design decisions"
metadata: 
  node_type: memory
  type: project
  originSessionId: a960d078-1290-4dc8-b6c6-73de9b4a5d38
---

Set 2 (fruit cubes) two-stage pipeline is implemented, mirroring Set 1 and kept fully
separate (own configs/models/datasets/runtime). See [[joon-robotics-competition-project]].

Key design decisions (non-obvious, worth keeping):
- Cube fruit faces are rendered as **thin textured decal quads** flush on 3 of 6 faces of a
  procedural white box (not STL+GeomSubsets — STL has no UVs). White-face-only `unknown`
  views then fall out of camera geometry naturally.
- The `fruit` vs `unknown` crop label is decided **analytically** in `sim/fruit_cube.py`
  (`CameraModel` + `fruit_visibility`): which fruit faces point at the camera and their
  projected area ratio — NOT from the renderer's bbox of the fruit. Model learns visible-fruit
  recognition, never hidden-cube guessing.
- Cubes are driven each frame via direct USD xform ops (like Set 1's camera); only the optional
  non-cube negatives use the Set-1-proven `rep.new_layer()` + `on_frame` trigger path.
- Decision policy is false-positive averse (wrong pickup −40): never picks `unknown`, REJECTs on
  ≥2 strong other-fruit votes, requests viewpoint change after 4 consecutive unknowns.

Open items: confirm real cube edge (`cubes.size_m`, default 0.06) and whether fruit-face layout
is fixed (`cubes.fixed_face_layout`); replace placeholder fruit textures in
`assets/fruit_textures/<fruit>/` with real photos. The Isaac generator is untestable in WSL/yolo
venv (needs Isaac python); decision policy + texture gen were verified locally.
