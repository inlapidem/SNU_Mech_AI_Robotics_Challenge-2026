"""Verify-gate + bin-capture state machine layered OVER the per-set decision policies.

Sensor architecture (v2, 2026-07): the RPLidar C1 sits at ~20-25 cm so its scan plane
passes above every object -- LiDAR is localization-only and this module takes NO object
candidates or distances from it. Two side Nuroum cams (role `search`) drive the existing
SEARCHING -> FAR_CANDIDATE -> TARGET_CONFIRMED progression through the unchanged per-set
policies; two front IMX219 cams (role `verify`) run the SAME detector+classifier but
their evidence feeds only the fusion rules here.

The robot has no gripper: a front bin (inner width `bin_width_m`) is pushed over the
object, funnel wings absorb small lateral error, and an IR sensor deep inside the bin
confirms deep seating. PICKUP_READY is therefore REDEFINED as CAPTURE_READY = "verify
gate passed + laterally aligned, safe to push in", and it can only be granted here,
from verify-role observations -- search cameras can never produce it.

Mission states (on top of the per-track policy states):
  SEARCH phase   SEARCHING / FAR_CANDIDATE / TARGET_CONFIRMED  (mirrors the search cams;
                 TARGET_CONFIRMED latches so a bad frame cannot un-confirm)
  VERIFY phase   VERIFYING        front cams gathering gate evidence
                 CAPTURE_READY    gate passed + aligned -> navigator may push in
                 VERIFY_REJECTED  veto: a front cam confidently saw a NON-target
  capture        BLIND_CAPTURE    object under the camera blind zone; hold heading,
                                  push, wait for IR. Camera observations are IGNORED
                                  here (the bin lip occludes the object; it has not
                                  "disappeared"). Veto no longer applies (spec: veto
                                  is valid only until BLIND_CAPTURE entry).
                 CAPTURE_MISSED   push limit expired without IR seating -> retreat
                 LOADED           IR confirmed deep seating; episode ends
                 OBJECT_LOST      IR lost the payload during transport -> re-search

IR integration: reading the IR hardware is the navigator's job; this class only offers
note_loaded(bool) / note_payload_lost() hooks (keyboard-simulatable in run_perception).

Alignment is pixel-ratio based (no camera calibration needed):
    allowed_offset_px = bbox_width_px * (bin_width_m - obj_width_m) / (2 * obj_width_m)
i.e. how far (in px) the bbox centre may sit off the camera axis while the object still
fits inside the bin, scaled by margin_factor for safety. Two fresh front cams must see
opposite-sign offsets with a small combined error; one cam falls back to |offset|.

False-positive-averse principle (wrong capture = -40, miss = 0) is preserved: every
promotion needs consecutive high-margin target frames, any confident non-target
observation vetoes, and ambiguity never leads to a capture.
"""

import time

from configs.merged_classes import set_of

# ---- mission states -------------------------------------------------------------
SEARCHING = "SEARCHING"
FAR_CANDIDATE = "FAR_CANDIDATE"
TARGET_CONFIRMED = "TARGET_CONFIRMED"
VERIFYING = "VERIFYING"
CAPTURE_READY = "CAPTURE_READY"
VERIFY_REJECTED = "VERIFY_REJECTED"
BLIND_CAPTURE = "BLIND_CAPTURE"
CAPTURE_MISSED = "CAPTURE_MISSED"
LOADED = "LOADED"
OBJECT_LOST = "OBJECT_LOST"

PHASE_SEARCH = "SEARCH"
PHASE_VERIFY = "VERIFY"

# ---- navigator requests ----------------------------------------------------------
REQ_REAPPROACH = "REAPPROACH"              # veto fired: back off, approach again
REQ_MICRO_ADJUST = "MICRO_ADJUST"          # persistent unknown: small viewpoint change
REQ_HOLD_HEADING_IR_WAIT = "HOLD_HEADING_IR_WAIT"  # blind push: keep heading, wait IR
REQ_RETREAT_RESEARCH = "RETREAT_RESEARCH"  # capture missed: back out, re-search
REQ_RESEARCH_NEARBY = "RESEARCH_NEARBY"    # payload lost in transport: search nearby

_SEARCH_RANK = {SEARCHING: 0, "UNKNOWN_CUBE": 0, "NON_TARGET_FRUIT": 0,
                FAR_CANDIDATE: 1, "TARGET_CANDIDATE": 2, TARGET_CONFIRMED: 3}


def allowed_offset_px(bbox_width_px, obj_width_m, bin_width_m):
    """Lateral bbox-centre offset (px) at which the object still enters the bin."""
    if obj_width_m <= 0:
        return 0.0
    return max(0.0, bbox_width_px * (bin_width_m - obj_width_m) / (2.0 * obj_width_m))


def object_width_m(cfg, cls):
    """Physical width (m) of an object by class name. The merged config lists every
    object under objects.real_size_m (shapes at their true printed size, fruit cubes at
    the 8 cm edge); the legacy per-set configs fall back to their own layout. Raises on
    an unknown class (fail fast at start)."""
    sizes = (cfg.get("objects") or {}).get("real_size_m", {})
    if cls in sizes:
        return float(sizes[cls])
    if cfg.get("set") == "set1":
        return float(cfg["objects"]["real_size_m"][cls])
    return float(cfg["cubes"]["size_m"])


class _VerifyCam:
    """Per-front-camera streaks + last selection (internal)."""

    def __init__(self):
        self.target_streak = 0
        self.veto_streak = 0
        self.veto_cls = None
        self.unknown_streak = 0
        self.offset_px = None
        self.bbox_width_px = None
        self.obj_width_m = None           # physical width of this cam's last selection
        self.last_seen_vu = -10**9        # verify-update counter at last selection
        self.last_strong_other_vu = -10**9

    def reset_streaks(self):
        self.target_streak = self.veto_streak = self.unknown_streak = 0
        self.veto_cls = None


class CaptureFSM:
    """Fuses per-camera pipeline results into one mission state. One instance per run.

    Call `update()` with every processed frame's results (any camera, any role);
    call `set_phase()/toggle_phase()` from the navigator or the keyboard toggle;
    call `note_loaded()/note_payload_lost()` from the IR integration.
    After a LOADED episode ends, `consume_reset()` returns True exactly once -- the
    caller must then reset the pipelines' trackers/votes (fresh episode).
    """

    def __init__(self, cfg, targets, clock=time.monotonic):
        self.cfg = cfg
        # Announced targets: a {set: name} dict (merged), a single name, or a list.
        if isinstance(targets, dict):
            names = [v for v in targets.values() if v]
        elif isinstance(targets, str):
            names = [targets]
        else:
            names = list(targets or [])
        self.targets = set(names)

        vf = cfg["verify"]
        rt = cfg["runtime"]
        # Split the verify + runtime blocks into a shared base + per-set overrides, then
        # build one flat table per set. A fruit crop must clear a stricter gate than a
        # shape crop, so verify_margin/veto_margin/conf_threshold are resolved by the
        # SELECTED object's derived set at evaluation time (see _update_verify).
        vbase = {k: v for k, v in vf.items() if k not in ("set1", "set2")}
        rshared = {k: v for k, v in rt.items() if k not in ("set1", "set2")}
        self.vf_set, self.conf_th_set = {}, {}
        for s in ("set1", "set2"):
            self.vf_set[s] = {**vbase, **vf.get(s, {})}
            self.conf_th_set[s] = float({**rshared, **rt.get(s, {})}.get(
                "conf_threshold", rshared.get("conf_threshold", 0.6)))
        self._width_cache = {}

        self.bin_width_m = float(vbase["bin_width_m"])
        self.verify_k = int(vbase["verify_k"])
        self.veto_m = int(vbase["veto_m"])
        self.margin_factor = float(vbase["margin_factor"])
        self.unknown_patience = int(vbase.get("verify_unknown_patience",
                                              rshared.get("unknown_patience", 4)))
        self.align_warn_min_px = float(vbase.get("align_warn_min_px", 15))
        self.pair_max_age = int(vbase.get("align_pair_max_age", 3))
        self.blind_bbox_px = float(vbase["blind_handoff_bbox_px"])
        self.blind_bottom_px = float(vbase.get("blind_handoff_bottom_px", 6))
        self.push_limit_s = float(vbase["capture_push_limit"])
        # Selection size range = union of the per-set ranges (per-object logic refines).
        ranges = [self.vf_set[s]["verify_bbox_px_range"] for s in ("set1", "set2")
                  if "verify_bbox_px_range" in self.vf_set[s]]
        self.bbox_range = (min(r[0] for r in ranges), max(r[1] for r in ranges))

        self.phase = PHASE_SEARCH
        self.state = SEARCHING
        self.payload_loaded = False
        self._clock = clock
        self._cams = {}                   # verify camera name -> _VerifyCam
        self._vu = 0                      # verify-update counter (freshness clock)
        self._blind_t0 = None
        self._needs_reset = False

        for w in self.startup_warnings():
            print(w)

    def _width(self, cls):
        if cls not in self._width_cache:
            self._width_cache[cls] = object_width_m(self.cfg, cls)
        return self._width_cache[cls]

    # ------------------------------------------------------------------ warnings
    def startup_warnings(self):
        """Targets whose bin clearance leaves almost no visual alignment allowance
        (e.g. the 13.6 cm octahedron in the 14 cm bin) are flagged once at start:
        vision cannot centre them better than the funnel wings can."""
        ref_w = self.bbox_range[1]
        warns = []
        for tgt in sorted(self.targets):
            w_m = self._width(tgt)
            allowed = allowed_offset_px(ref_w, w_m, self.bin_width_m)
            if allowed < self.align_warn_min_px:
                warns.append(
                    f"[capture] WARNING: target '{tgt}' ({w_m*100:.1f} cm) in the "
                    f"{self.bin_width_m*100:.1f} cm bin allows only {allowed:.1f}px of "
                    f"lateral offset even at a {ref_w}px bbox "
                    f"(< {self.align_warn_min_px:.0f}px). Visual alignment margin is "
                    f"nearly zero -- capture relies on the funnel wings.")
        return warns

    # ------------------------------------------------------------------ phase API
    def set_phase(self, phase):
        """Navigator-facing phase switch. Entering VERIFY starts a FRESH gate
        (streaks cleared); entering SEARCH also acknowledges/clears the sticky
        VERIFY_REJECTED / CAPTURE_MISSED / OBJECT_LOST outcome states."""
        if phase not in (PHASE_SEARCH, PHASE_VERIFY):
            raise ValueError(f"phase must be {PHASE_SEARCH}|{PHASE_VERIFY}")
        if phase == self.phase and self.state not in (VERIFY_REJECTED, CAPTURE_MISSED,
                                                      OBJECT_LOST, LOADED):
            return self.state
        self.phase = phase
        for c in self._cams.values():
            c.reset_streaks()
        self._blind_t0 = None
        if self.state not in (LOADED,):
            self.state = VERIFYING if phase == PHASE_VERIFY else SEARCHING
        return self.state

    def toggle_phase(self):
        return self.set_phase(PHASE_VERIFY if self.phase == PHASE_SEARCH
                              else PHASE_SEARCH)

    # ------------------------------------------------------------------ IR hooks
    def note_loaded(self, seated):
        """IR seating report (navigator owns the hardware read; tests use the keyboard).

        seated=True  -> LOADED: episode over; trackers/votes must be reset
                        (consume_reset()) and the FSM returns to SEARCH.
        seated=False -> only meaningful once a payload is loaded (transport check):
                        equivalent to note_payload_lost(). While pushing, a False
                        reading just means "not seated yet" and is ignored (the
                        capture_push_limit timer handles a stuck push)."""
        if seated:
            self.payload_loaded = True
            self.state = LOADED
            self._blind_t0 = None
            self._needs_reset = True
            self.phase = PHASE_SEARCH
            for c in self._cams.values():
                c.reset_streaks()
            return self.state
        if self.payload_loaded:
            return self.note_payload_lost()
        return self.state

    def note_payload_lost(self):
        """IR reports the bin empty during transport -> search near the current pose."""
        self.payload_loaded = False
        self.state = OBJECT_LOST
        self._blind_t0 = None
        return self.state

    def consume_reset(self):
        """True exactly once after LOADED: caller resets the pipelines' tracking."""
        if self._needs_reset:
            self._needs_reset = False
            return True
        return False

    # ------------------------------------------------------------------ main update
    def update(self, camera, role, results, frame_size, now=None):
        """Fuse one processed frame. `results` is the pipeline's per-detection list,
        `frame_size` is (H, W). Returns the navigator-facing dict (state, events,
        request, and -- for verify cams in the VERIFY phase -- per-frame steering)."""
        now = self._clock() if now is None else now
        events = []
        request = None

        # Blind-push timeout is checked on EVERY update so a throttled camera set
        # cannot stall the CAPTURE_MISSED transition.
        if self.state == BLIND_CAPTURE:
            if now - self._blind_t0 > self.push_limit_s:
                self.state = CAPTURE_MISSED
                events.append("CAPTURE_MISSED")
            else:
                request = REQ_HOLD_HEADING_IR_WAIT
        # Sticky outcome states keep re-emitting their request until the navigator
        # acknowledges via set_phase() (or the keyboard toggle in bench tests).
        if self.state == CAPTURE_MISSED:
            request = REQ_RETREAT_RESEARCH
        elif self.state == VERIFY_REJECTED:
            request = REQ_REAPPROACH

        out = {"camera": camera, "role": role, "phase": self.phase,
               "state": self.state, "events": events, "request": request,
               "payload_loaded": self.payload_loaded,
               "steering": None, "verify": None}

        if self.state == OBJECT_LOST:
            out["request"] = REQ_RESEARCH_NEARBY
            return out
        if self.state in (BLIND_CAPTURE, CAPTURE_MISSED):
            # Camera observations must NOT move the state here: the object is (or
            # was) inside the bin's blind zone, not gone.
            return out
        if self.state == LOADED:
            # Episode over (reported once): resume searching for the next object while
            # the navigator transports the payload (payload_loaded stays True so
            # note_loaded(False)/note_payload_lost() still means OBJECT_LOST).
            self.state = SEARCHING

        if role == "search":
            self._update_search(results)
        else:
            self._update_verify(camera, results, frame_size, now, events, out)
        out["state"] = self.state
        return out

    # ------------------------------------------------------------------ search side
    def _update_search(self, results):
        if self.phase != PHASE_SEARCH or self.state not in (SEARCHING, FAR_CANDIDATE,
                                                            TARGET_CONFIRMED):
            return
        best = max((_SEARCH_RANK.get(r["state"], 0) for r in results), default=0)
        if best >= 3:
            self.state = TARGET_CONFIRMED          # latches until phase change/veto
        elif self.state != TARGET_CONFIRMED:
            self.state = FAR_CANDIDATE if best >= 1 else SEARCHING

    # ------------------------------------------------------------------ verify side
    def _select(self, results, frame_size, strict):
        """Verification-target prior: the box nearest the image centre whose width is
        inside verify_bbox_px_range (LiDAR distance is deliberately not used). With
        strict=False (CAPTURE_READY push) the size range is dropped: the object may
        legitimately outgrow the range right before the blind handoff."""
        H, W = frame_size
        best, best_d = None, None
        for r in results:
            x0, y0, x1, y1 = r["bbox"]
            w = x1 - x0
            if strict and not (self.bbox_range[0] <= w <= self.bbox_range[1]):
                continue
            d = ((x0 + x1) / 2 - W / 2) ** 2 + ((y0 + y1) / 2 - H / 2) ** 2
            if best_d is None or d < best_d:
                best, best_d = r, d
        return best

    def _cam(self, camera):
        if camera not in self._cams:
            self._cams[camera] = _VerifyCam()
        return self._cams[camera]

    def _steering(self, camera, frame_size):
        """Per-frame visual-servoing feedback for the navigator (spec rule 8)."""
        per_cam, fresh = {}, []
        for name, c in self._cams.items():
            if c.offset_px is None or c.obj_width_m is None:
                continue
            age = self._vu - c.last_seen_vu
            per_cam[name] = {"offset_px": round(c.offset_px, 1),
                             "bbox_width_px": round(c.bbox_width_px, 1),
                             "allowed_offset_px": round(allowed_offset_px(
                                 c.bbox_width_px, c.obj_width_m, self.bin_width_m), 1),
                             "age": age}
            if age <= self.pair_max_age:
                fresh.append(c)
        aligned, combined, allowed, pair = False, None, 0.0, False
        if len(fresh) >= 2:
            a, b = sorted(fresh, key=lambda c: self._vu - c.last_seen_vu)[:2]
            pair = True
            combined = (a.offset_px + b.offset_px) / 2.0
            allowed = allowed_offset_px((a.bbox_width_px + b.bbox_width_px) / 2.0,
                                        (a.obj_width_m + b.obj_width_m) / 2.0,
                                        self.bin_width_m)
            opposite = a.offset_px * b.offset_px <= 0
            aligned = opposite and abs(combined) <= allowed * self.margin_factor
        elif len(fresh) == 1:
            c = fresh[0]
            combined = c.offset_px
            allowed = allowed_offset_px(c.bbox_width_px, c.obj_width_m,
                                        self.bin_width_m)
            aligned = abs(combined) <= allowed * self.margin_factor
        return {"per_cam": per_cam,
                "combined_offset_px": None if combined is None else round(combined, 1),
                "allowed_offset_px": round(allowed, 1),
                "margin_factor": self.margin_factor,
                "pair": pair, "aligned": aligned}

    def _update_verify(self, camera, results, frame_size, now, events, out):
        cam = self._cam(camera)
        self._vu += 1
        H, W = frame_size

        in_gate = self.state in (VERIFYING, CAPTURE_READY) and self.phase == PHASE_VERIFY
        sel = self._select(results, frame_size, strict=self.state != CAPTURE_READY)

        if sel is None:
            cam.target_streak = cam.veto_streak = 0
            cam.offset_px = None
            if results:               # something is there but nothing verifiable
                cam.unknown_streak += 1
        else:
            x0, y0, x1, y1 = sel["bbox"]
            cam.offset_px = (x0 + x1) / 2.0 - W / 2.0
            cam.bbox_width_px = x1 - x0
            cam.last_seen_vu = self._vu
            cls, conf, margin = sel["cls"], sel["conf"], sel["margin"]
            # Resolve the gate by the selected object's derived set (fruit gates are
            # stricter). 'unknown'/None classes never pass strong_target/other anyway.
            s = set_of(cls) or "set2"
            conf_th = self.conf_th_set[s]
            verify_margin = self.vf_set[s].get("verify_margin", 0.2)
            veto_margin = self.vf_set[s].get("veto_margin", 0.35)
            if cls not in (None, "unknown"):
                cam.obj_width_m = self._width(cls)
            strong_target = (cls in self.targets and conf >= conf_th
                             and margin >= verify_margin)
            # A non-target 'cube' is AMBIGUOUS (a plain cube OR a target fruit cube's
            # blank side), so it must NOT veto -- else a target fruit whose fruit faces
            # are momentarily hidden gets rejected/blacklisted. It counts as unknown
            # (unknown_streak -> micro viewpoint adjust). Only a definite non-target
            # (a specific wrong fruit, or an unambiguous non-cube shape) vetoes.
            ambiguous_cube = cls == "cube" and "cube" not in self.targets
            strong_other = (cls not in (None, "unknown") and cls not in self.targets
                            and not ambiguous_cube
                            and conf >= conf_th and margin >= veto_margin)
            cam.target_streak = cam.target_streak + 1 if strong_target else 0
            cam.veto_streak = cam.veto_streak + 1 if strong_other else 0
            cam.veto_cls = cls if strong_other else (cam.veto_cls if cam.veto_streak else None)
            if strong_other:
                cam.last_strong_other_vu = self._vu
            cam.unknown_streak = 0 if (strong_target or strong_other) \
                else cam.unknown_streak + 1
            out["verify"] = {"selected_track": sel.get("track"),
                             "cls": cls, "conf": round(conf, 3),
                             "margin": round(margin, 3),
                             "target_streak": cam.target_streak,
                             "veto_streak": cam.veto_streak,
                             "unknown_streak": cam.unknown_streak}

        if not in_gate:
            return

        steering = self._steering(camera, frame_size)
        out["steering"] = steering

        # ---- veto: valid in VERIFYING *and* CAPTURE_READY, i.e. right up to the
        # BLIND_CAPTURE handoff. Pushing stops if the front cams see a non-target.
        if cam.veto_streak >= self.veto_m:
            self.state = VERIFY_REJECTED
            events.append(f"VETO:{cam.veto_cls}")
            out["request"] = REQ_REAPPROACH
            for c in self._cams.values():
                c.reset_streaks()
            return

        if self.state == VERIFYING:
            # ---- persistent unknown -> reuse the re-observe idea: request a micro
            # viewpoint adjustment instead of giving up (spec rule 6).
            if cam.unknown_streak >= self.unknown_patience \
                    and cam.unknown_streak % self.unknown_patience == 0:
                out["request"] = REQ_MICRO_ADJUST
                events.append("UNKNOWN_PERSISTS")
            # ---- promotion: K consecutive high-margin target frames on THIS cam,
            # no recent confident non-target on ANY OTHER front cam, and aligned.
            others_clean = all(self._vu - c.last_strong_other_vu > self.verify_k
                               for n, c in self._cams.items() if n != camera)
            if (cam.target_streak >= self.verify_k and others_clean
                    and steering["aligned"]):
                self.state = CAPTURE_READY
                events.append("CAPTURE_READY")

        if self.state == CAPTURE_READY and sel is not None:
            # ---- blind-zone handoff: bbox bottom at the frame border or bbox tall
            # enough means the object is at the bin lip; cameras go blind from here.
            x0, y0, x1, y1 = sel["bbox"]
            if y1 >= H - self.blind_bottom_px or (y1 - y0) >= self.blind_bbox_px:
                self.state = BLIND_CAPTURE
                self._blind_t0 = now
                events.append("BLIND_CAPTURE")
                out["request"] = REQ_HOLD_HEADING_IR_WAIT
