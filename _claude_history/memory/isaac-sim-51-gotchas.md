---
name: isaac-sim-51-gotchas
description: "Hard-won fixes for Isaac Sim 5.1 Replicator synthetic-data generation (black frames, lights, run loop)"
metadata: 
  node_type: memory
  type: reference
  originSessionId: c1b83ff7-86ff-4467-adfb-c2717d5ef135
---

Isaac Sim **5.1** (Omniverse standalone at `C:\Users\user\Documents\IsaacSim\python.bat`) Replicator gotchas, discovered while building the polyhedra data generator. All encoded in `isaac/generate_replicator.py` and `sim/generate_set1_data.py`:

- **`rep.create.light` produces NO illumination** in this build → renders come out pure black. Fix: create lights with the USD API directly (`UsdLux.DistantLight`/`DomeLight` on the stage) and randomize their intensity via `.GetIntensityAttr().Set(...)`.
- **RGB capture is black unless the renderer accumulates** → drive frames with `rep.orchestrator.step(rt_subframes=32)` in a Python loop, NOT `rep.orchestrator.run()` + `simulation_app.update()` pumping (that gives 1 sample → black).
- **`rgb` annotator may return float [0,1]** → must `*255 .astype(uint8)` before saving or the PNG is black. (Also add a mean-brightness gate to drop occasional glitch frames.)
- `asset_converter.create_converter_context()` was removed → use `AssetConverterContext()`.
- `trigger.on_frame(num_frames=...)` warns it's deprecated for `max_execs` (still works).
- **Windows CMD (`python.bat`) can't use a `\\wsl.localhost\...` UNC cwd** → copy the repo to `C:\joon` first, generate there, robocopy results back to WSL.
- STL→USD: normalize to a known size with op order `[Translate(-center), Scale]` (scale-about-center); the reverse order mis-places the mesh. Our 4 solids normalize to 0.2 m max-extent.

See [[joon-robotics-competition-project]].
