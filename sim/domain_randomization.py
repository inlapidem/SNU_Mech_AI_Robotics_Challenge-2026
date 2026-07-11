"""Domain-randomization helpers: lighting, colour temperature, robot pose sampling.

Lights are created/edited via the USD API (rep.create.light produced no illumination
in this Isaac build — see isaac/generate_replicator.py history). Object pose and the
white-plastic material are randomized with Replicator distributions in the generator's
trigger; this module covers the per-frame, Python-driven pieces.

Shared infrastructure (Set-agnostic).
"""

import math

from pxr import UsdLux, UsdGeom, Sdf, Gf


def create_lights(stage, root="/World/Lights"):
    """A distant 'sun' + a dome fill. Returns prims for per-frame randomization."""
    sun = UsdLux.DistantLight.Define(stage, Sdf.Path(root + "/Sun"))
    sun.CreateIntensityAttr(4000.0)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 0.0))

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path(root + "/Dome"))
    dome.CreateIntensityAttr(1200.0)
    return {"sun": sun, "dome": dome}


def kelvin_to_rgb(kelvin):
    """Approximate colour-temperature -> linear RGB (0..1). Tanner Helland fit."""
    t = max(1000.0, min(40000.0, kelvin)) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        r = 329.698727446 * ((t - 60) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60) ** -0.0755148492)
    if t >= 66:
        b = 255.0
    elif t <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10) - 305.0447927307
    clamp = lambda v: max(0.0, min(255.0, v)) / 255.0
    return (clamp(r), clamp(g), clamp(b))


def randomize_lights(lights, light_cfg, rng):
    """Set sun/dome intensity, sun direction, and colour temperature for one frame."""
    sun, dome = lights["sun"], lights["dome"]
    sun.GetIntensityAttr().Set(float(rng.uniform(*light_cfg["sun_intensity"])))
    dome.GetIntensityAttr().Set(float(rng.uniform(*light_cfg["dome_intensity"])))

    pitch = rng.uniform(*light_cfg["sun_pitch_deg"])
    yaw = rng.uniform(*light_cfg["sun_yaw_deg"])
    UsdGeom.Xformable(sun).GetOrderedXformOps()[0].Set(Gf.Vec3f(float(pitch), 0.0, float(yaw)))

    rgb = kelvin_to_rgb(rng.uniform(*light_cfg["color_temp"]))
    sun.GetColorAttr().Set(Gf.Vec3f(*rgb))
    dome.GetColorAttr().Set(Gf.Vec3f(*rgb))


def sample_jitter(robot_cfg, rng):
    """Per-frame camera mount-error (height/pitch/yaw/roll)."""
    j = robot_cfg["jitter"]
    return {
        "height": float(rng.uniform(*j["height"])),
        "pitch_deg": float(rng.uniform(*j["pitch_deg"])),
        "yaw_deg": float(rng.uniform(*j["yaw_deg"])),
        "roll_deg": float(rng.uniform(*j["roll_deg"])),
    }


def sample_robot_base(rng, dist_range=(0.6, 3.0)):
    """Robot base_link pose on a ring around the object region (origin), facing it.

    Returns (base_xy, base_yaw_deg). Distance varies so the model sees objects from
    far (detect) to near (classify); base_yaw points the robot at the region centre.
    """
    ang = rng.uniform(-math.pi, math.pi)
    dist = rng.uniform(*dist_range)
    base_xy = (dist * math.cos(ang), dist * math.sin(ang))
    base_yaw_deg = math.degrees(ang) + 180.0          # face the origin
    return base_xy, base_yaw_deg, dist


# ---------------------------------------------------------------- long-range sampling
def sample_arena_offset(rng, scfg, arena_half=2.0, cluster_r=0.3):
    """Arena translation for this frame (the object cluster stays at the origin).

    With prob wall_contact_frac the nearest wall is pulled to wall_gap_m of the
    cluster centre (white object against bright wood wall, touching allowed);
    otherwise the cluster lands uniformly anywhere with full wall clearance.
    """
    if rng.uniform() < scfg.get("wall_contact_frac", 0.0):
        gap = rng.uniform(*scfg.get("wall_gap_m", [cluster_r + 0.03, 0.6]))
        side = rng.randint(4)                      # which wall hugs the cluster: N,S,E,W
        mag = arena_half - gap                     # wall plane at +/-arena_half + offset
        along = rng.uniform(-(arena_half - cluster_r - 0.2), arena_half - cluster_r - 0.2)
        ox, oy = ((along, -mag), (along, mag), (-mag, along), (mag, along))[side]
    else:
        lim = arena_half - cluster_r - 0.15
        ox, oy = rng.uniform(-lim, lim), rng.uniform(-lim, lim)
    return float(ox), float(oy)


def _max_dist_along(ang, arena_offset, arena_half, margin):
    """Max camera distance from the cluster (origin) along `ang` that keeps the eye
    inside the (offset) arena. eye = dist * (cos ang, sin ang); arena spans
    offset +/- arena_half per axis."""
    ox, oy = arena_offset
    max_d = 10.0
    for d, o in ((math.cos(ang), ox), (math.sin(ang), oy)):
        if abs(d) > 1e-6:
            for bound in (o - arena_half + margin, o + arena_half - margin):
                t = bound / d
                if t > 0:
                    max_d = min(max_d, t)
    return max_d


def sample_camera_view(rng, scfg, arena_offset, arena_half=2.0, margin=0.15):
    """(approach angle, camera distance, far?) for this frame: near/far mixture.

    The angle is re-sampled a few times looking for a direction with enough room
    for the drawn distance. For far draws this defeats wall clipping (cluster
    against a wall -> viewed from across the arena, 3.5 m+; otherwise the far tail
    collapses onto ~2 m). For near draws it prevents the opposite failure: in a
    wall-contact frame an angle toward the pulled-in wall leaves only ~0.15 m of
    room, which would put the camera INSIDE the object cluster.
    """
    rng_near = scfg.get("cam_dist_near", [0.4, 1.5])
    rng_far = scfg.get("cam_dist_far", [1.5, 3.8])
    far = rng.uniform() < scfg.get("far_frac", 0.45)
    dist = rng.uniform(*(rng_far if far else rng_near))
    ang = rng.uniform(-math.pi, math.pi)
    best_ang, best_d = ang, _max_dist_along(ang, arena_offset, arena_half, margin)
    for _ in range(8):
        if best_d >= dist:
            break
        ang = rng.uniform(-math.pi, math.pi)
        d = _max_dist_along(ang, arena_offset, arena_half, margin)
        if d > best_d:
            best_ang, best_d = ang, d
    return float(best_ang), float(min(dist, best_d)), far
