#!/usr/bin/env python3
"""전체 미션 WSL 모의 검증 (ROS/하드웨어 불필요) — sim_localizer.py 와 같은 취지.

3분 경기를 통째로 시뮬레이션한다: 50cm 격자 42지점에 세트1(16)+세트2(12) 랜덤
배치, 차동구동 운동학, 카메라 관측(측면 search 2 + 전면 verify 2, FOV/거리 제한,
세트2 과일면 가시성은 접근 방향에 따라 랜덤), 빈 포획 물리(정렬돼야 안착),
IR 안착 센서, 보관함 하역. 미션 로직(mission_fsm)이 실제 코드 그대로 돈다.

실행:
  yolo/bin/python navigation/sim_mission.py            # 시나리오 전체 PASS 확인
  yolo/bin/python navigation/sim_mission.py --seeds 20 # 명목 시나리오 시드 수 늘리기

시나리오:
  A nominal      랜덤 배치 x seeds — 오픽업 0, 벽충돌 0, 평균 점수 리포트
  B ir-dropout   운반 중 IR 순간 끊김(0.4s) — 오탐 하역중단 없이 완주
  C payload-lost 운반 중 실제 유실 1회 — 재탐색/재포획 복구
  D capture-miss 첫 푸시 실패(빗맞음) — 재시도 로직
  E endgame      잔여 60초 시작 — 시간 컷오프(hail_mary off): 종료 시 적재 방치 금지
  E2 hail-mary   같은 60초, hail_mary on(기본) — 막판 컷오프 무시 시도가 안전하고
                 컷오프-only 대비 점수가 안 떨어지는지 (종료 적재는 감점 0 이라 허용)
  F loc-degraded 20~35초 위치추정 불량 — 정지 대기 후 재개, 완주
"""

import argparse
import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nav_core import ArenaGeometry, wrap_angle
from mission_fsm import (MissionFSM, PerceptionFrame, TOUR, GOTO, APPROACH,
                         CAPTURE, TRANSPORT, DEPOSIT_PUSH, DEPOSIT_RELEASE,
                         PARK, DONE, REQ_MICRO_ADJUST,
                         FSM_CAPTURE_READY, FSM_BLIND_CAPTURE,
                         FSM_VERIFY_REJECTED, FSM_LOADED)

SHAPES = ["cube", "octa", "dodeca", "icosa"]
FRUITS = ["apple", "orange", "banana", "pineapple"]
DT = 0.05

# 과일 배치 모델: "opposite2"(현행 룰: 윗면+반대편 옆2면) | "random3"(구 비교용)
FRUIT_LAYOUT = "opposite2"

# 무지면 큐브 관측이 싣고 오는 set 라벨. 통합 엔진(merged_pipeline)은 set_of("cube")
# ="set1" 로 라우팅하므로 실전 계약은 "set1". 내비게이션은 이 라벨을 무시하고
# 클래스로부터 set 을 재유도해야 정상 — None(구 per-set 방출)과 결과가 동일해야
# 한다. "set1" 로 두는 것이 통합 엔진 최악 입력 재현.
CUBE_SIGHTING_SET = "set1"

# ------------------------------------------------------------------ 경기장 생성


def make_arena(rng):
    """42개 배치 지점(7x6, 50cm 간격) 중 스타트존 1m 이내 제외 후 28개 배치."""
    pts = [(0.5 + 0.5 * i, 0.75 + 0.5 * j) for i in range(7) for j in range(6)]
    start = (3.8, 0.2)
    pts = [p for p in pts if math.hypot(p[0] - start[0], p[1] - start[1]) >= 1.0]
    rng.shuffle(pts)
    objs = []
    k = 0
    for shape in SHAPES:
        for _ in range(4):
            objs.append(dict(id=k, set="set1", cls=shape, x=pts[k][0],
                             y=pts[k][1], payload=False, done=False))
            k += 1
    for fruit in FRUITS:
        for _ in range(3):
            yaw = rng.uniform(0, 2 * math.pi)
            if FRUIT_LAYOUT == "random3":
                # 구 룰(비교용): 6면 중 3면 랜덤, 옆면만 판독. 커버리지 편차 큼.
                faces = rng.sample(range(6), 3)
                normals = [yaw + f * math.pi / 2 for f in faces if f < 4]
            else:
                # 룰(2026-07-15): 과일 사진은 윗면 + 옆면 '서로 반대편 2면' —
                # 어느 방향에서 접근해도 웬만하면 과일이 보이도록 세팅. 카메라가
                # 낮아 윗면은 판독 불가로 두고, 반대편 옆 2면(±180°)이 탐지 채널.
                # 두 면이 반대편이라 옆면 가시 260°(72%), 무지 사각은 과일축 수직
                # 방향 두 50° 창뿐. 큐브 방향(yaw)·과일축 선택은 랜덤.
                axis = yaw + rng.choice([0.0, math.pi / 2])
                normals = [axis, axis + math.pi]
            objs.append(dict(id=k, set="set2", cls=fruit, x=pts[k][0],
                             y=pts[k][1], payload=False, done=False,
                             fruit_normals=normals))
            k += 1
    return objs


def fruit_visible(obj, from_angle):
    """관측 방향(물체→로봇)에서 과일 옆면이 보이는가 — 면 법선 ±65° 가시.

    새 룰에서 옆 과일은 반대편 2면이라 가시 커버리지 260°. 무지 뷰가 시선
    ±45° 내 법선의 면을 '무지'로 인증하려면(내비게이터 무지면 증명의 건전성)
    가시 한계가 45°보다 넓어야 한다: 65° > 45° ✓. 반대편 2면 배치이므로
    set2 큐브의 무지 사각은 항상 180° 떨어진 두 50° 창 → 그 두 창에만 갇힌
    관측은 연속 갭 130°를 남겨 set1 오인증이 불가능하다(증명은 _maybe_confirm).
    """
    if obj["set"] != "set2":
        return True
    return any(abs(wrap_angle(from_angle - n)) < math.radians(65)
               for n in obj["fruit_normals"])


# ------------------------------------------------------------------ 모의 인지

class SimPerception:
    """search/verify 캠 관측 + capture_fsm 수준의 상태/요청을 흉내낸다.

    실제 인지의 보수성(연속 프레임 요구·veto·unknown 시 MICRO_ADJUST)을 그대로
    재현하되 프레임 단위 디테일은 생략. 미션이 보내는 cmd(set_phase 등)를 반영.
    """
    SEARCH_R, SEARCH_FOV = 2.6, math.radians(90)
    CLS_R = 1.5                       # 이 안쪽이어야 search 캠이 분류 가능
    VERIFY_R, VERIFY_FOV = 1.3, math.radians(62)
    VERIFY_BLIND = 0.28

    def __init__(self, objs, targets, rng):
        self.objs = objs
        self.targets = targets
        self.rng = rng
        self.phase = "SEARCH"
        self.loaded = False
        self.fsm_state = "SEARCHING"
        self.hits = {}                # id -> search 관측 누적
        self.v_streak = {}            # id -> verify 연속 관측
        self.u_streak = 0             # unknown 연속
        self.request = None

    def apply_cmds(self, cmds):
        for c in cmds:
            if c["cmd"] == "set_phase":
                self.phase = c["phase"]
                self.v_streak.clear()
                self.u_streak = 0
                self.request = None
                # 페이즈 전환은 판정 재무장: 이전 에피소드의 READY/REJECTED 파기
                self.fsm_state = ("SEARCHING" if self.phase == "SEARCH"
                                  else "VERIFYING")
            elif c["cmd"] == "note_loaded":
                self.loaded = c["loaded"]
                self.fsm_state = FSM_LOADED if c["loaded"] else self.fsm_state
            elif c["cmd"] == "note_payload_lost":
                # capture_fsm: LOADED -> OBJECT_LOST, 관측 재개 (loaded 해제)
                self.loaded = False
                self.fsm_state = "OBJECT_LOST"
                self.v_streak.clear()
            elif c["cmd"] == "reset_tracking":
                self.hits.clear()
                self.v_streak.clear()

    def _is_target(self, o):
        return o["cls"] == self.targets.get(o["set"])

    def frame(self, pose, rng):
        x, y, yaw = pose
        sightings = []
        # --- 측면 search 캠 2대 (±90°) ---
        for cam_yaw in (math.pi / 2, -math.pi / 2):
            for o in self.objs:
                if o["done"] or o["payload"]:
                    continue
                b, r, ang = self._see(pose, o, cam_yaw,
                                      self.SEARCH_FOV, self.SEARCH_R)
                if b is None:
                    continue
                self.hits[o["id"]] = self.hits.get(o["id"], 0) + 1
                h = self.hits[o["id"]]
                cls = None
                sset = None
                state = "SEARCHING"
                if h >= 3:
                    state = "FAR_CANDIDATE"
                if r < self.CLS_R and h >= 5:
                    if o["set"] == "set1" and o["cls"] != "cube":
                        cls, sset = o["cls"], "set1"   # 다면체는 형상으로 분류
                    elif o["set"] == "set2" and fruit_visible(o, ang):
                        cls, sset = o["cls"], "set2"   # 과일면이 이쪽을 향함
                    else:
                        # 큐브류 무지면 뷰: 통합 엔진은 set_of("cube")="set1" 로
                        # 라우팅하므로 set="set1" 을 실어 보내고, cube 타깃 경기면
                        # set1 정책이 votes 후 TARGET_CONFIRMED 까지 낼 수 있다
                        # (최악 입력). 내비게이션이 이 set/state 를 무시하고 다시점
                        # 인증만 신뢰하는지가 통합 견고성의 핵심 — sim 이 그 최악
                        # 입력을 재현한다.
                        cls, sset = "cube", CUBE_SIGHTING_SET
                        state = "CUBE_BLANK_VIEW"
                        if self.targets.get("set1") == "cube" and h >= 5:
                            state = "TARGET_CONFIRMED"
                    if cls != "cube":
                        state = ("TARGET_CONFIRMED" if self._is_target(o)
                                 else "NON_TARGET")
                sightings.append(dict(
                    set=sset, cls=cls, state=state,
                    bearing=b + rng.gauss(0, 0.01),
                    range=r * (1 + rng.gauss(0, 0.05))))

        # --- 전면 verify 캠 (미션이 VERIFY 페이즈로 바꿨을 때만 게이트 동작) ---
        # 주의: 적재(loaded) 중에도 verify 는 돈다 — 더블 캐리의 2번째 물체 검증.
        # 실제 capture_fsm 은 LOADED 로 에피소드가 끝나는 1물체 모델이라, 실물
        # 적용 시 '적재 상태 2번째 VERIFY 에피소드' 지원이 필요하다 (README).
        steering = None
        verify_range = None
        verify_bearing = None
        if self.phase == "VERIFY":
            best = None
            for o in self.objs:
                if o["done"] or o["payload"]:
                    continue
                b, r, ang = self._see(pose, o, 0.0, self.VERIFY_FOV,
                                      self.VERIFY_R)
                if b is None:
                    continue
                if best is None or r < best[1]:
                    best = (o, r, b, ang)
            if best is not None and best[1] > self.VERIFY_BLIND:
                o, r, b, ang = best
                verify_range = r * (1 + rng.gauss(0, 0.06))
                verify_bearing = b + rng.gauss(0, 0.01)
                lat = r * math.sin(b)   # +lat = 물체가 로봇 왼쪽
                allowed_m = 0.03        # (bin 0.14 - obj 0.08)/2
                # 픽셀 관례: 이미지 오른쪽 = +offset. 왼쪽 물체(+lat)는 -offset.
                steering = dict(combined_offset_px=-lat * 1000.0,
                                allowed_offset_px=allowed_m * 1000.0,
                                aligned=abs(lat) < allowed_m * 0.7)
                sid = o["id"]
                self.v_streak[sid] = self.v_streak.get(sid, 0) + 1
                identifiable = (o["set"] == "set1" and o["cls"] != "cube") or \
                    (o["set"] == "set2" and fruit_visible(o, ang))
                # verify 캠 관측도 sighting 으로 전달 (실제 UDP 브리지와 동일) —
                # 접근 중 기억 속 물체 위치가 근거리 관측으로 보정된다
                if identifiable:
                    v_cls, v_set = o["cls"], o["set"]
                    v_state = ("TARGET_CONFIRMED" if self._is_target(o)
                               else "NON_TARGET")
                else:
                    v_cls, v_set, v_state = "cube", CUBE_SIGHTING_SET, "CUBE_BLANK_VIEW"
                sightings.append(dict(
                    set=v_set, cls=v_cls, state=v_state,
                    bearing=b + rng.gauss(0, 0.01),
                    range=r * (1 + rng.gauss(0, 0.04))))
                if identifiable:
                    self.u_streak = 0
                    if self._is_target(o):
                        if self.v_streak[sid] >= 6 and steering["aligned"]:
                            self.fsm_state = FSM_CAPTURE_READY
                        elif self.fsm_state != FSM_CAPTURE_READY:
                            self.fsm_state = "VERIFYING"
                    elif self.v_streak[sid] >= 4:
                        self.fsm_state = FSM_VERIFY_REJECTED
                elif self.targets.get("set1") == "cube":
                    # cube 공지 경기: verify 는 무지면 큐브를 'cube'로 승인할
                    # 수밖에 없다 (세트2 큐브와 단일 시점 구분 불가). 오픽업
                    # 방어는 내비게이터의 4사분면 무지면 증명이 담당한다.
                    self.u_streak = 0
                    if self.v_streak[sid] >= 6 and steering["aligned"]:
                        self.fsm_state = FSM_CAPTURE_READY
                    elif self.fsm_state != FSM_CAPTURE_READY:
                        self.fsm_state = "VERIFYING"
                else:
                    self.u_streak += 1
                    self.fsm_state = "VERIFYING"
                    if self.u_streak > 40:      # ~2초 unknown 지속
                        self.request = REQ_MICRO_ADJUST
                        self.u_streak = 0
            elif best is not None:              # 블라인드 존 — 푸시 구간
                if self.fsm_state == FSM_CAPTURE_READY:
                    self.fsm_state = FSM_BLIND_CAPTURE

        f = PerceptionFrame(fsm_state=self.fsm_state, request=self.request,
                            steering=steering, verify_range=verify_range,
                            verify_bearing=verify_bearing, sightings=sightings)
        self.request = None
        return f

    def _see(self, pose, o, cam_yaw, fov, max_r):
        dx, dy = o["x"] - pose[0], o["y"] - pose[1]
        r = math.hypot(dx, dy)
        if r < 0.05 or r > max_r:
            return None, None, None
        world = math.atan2(dy, dx)
        b = wrap_angle(world - pose[2])                # 로봇 기준 방위
        if abs(wrap_angle(b - cam_yaw)) > fov / 2:
            return None, None, None
        return b, r, wrap_angle(world + math.pi)      # 물체가 로봇을 보는 각


# ------------------------------------------------------------------ 물리 세계

class World:
    BODY_R = 0.17         # 충돌용 원 근사 (robot.stl 34×31cm; 내접~0.155/외접~0.23)
    OBJ_R = 0.05
    POCKET = 0.07         # 로봇 중심 -> 스쿱 안착점 (3MF: 바스켓 깊이140·IR 내벽50mm)
    FRONT_OFF = 0.35      # 로봇 중심 -> 입구에 문 2번째 물체 중심 (더블캐리 off 시 미사용)
    CAPTURE_LAT = 0.05    # 포획 허용 횡오차 (깔때기 날개 포함)
    # 전방 스쿱 개구부: 포켓(0.10)이 BODY_R(0.17) 안이라, 몸체 원으로 치면 물체가
    # 포획 전에 밀려난다. 실제론 앞이 뚫린 U채널이므로 이 콘 안의 물체는 몸체
    # 밀침에서 제외(개구부로 진입 → 포획 대상). 채널 반폭 ~0.06, 전방 도달 ~0.24.
    SCOOP_HALF = 0.06
    SCOOP_REACH = 0.24
    # --- 적재구역 혼잡 모델 ---
    STORE_X = 0.45        # 이 x 안쪽에서 물체를 놓으면 '하역'으로 간주 (경기장 유실과 구분)
    STORE_Y = 0.55
    STORE_MAX = 0.36      # 경계 완전 내부 상한 (score 와 일치); 이보다 밖이면 스필=미채점
    PILE_PITCH = 0.09     # 8cm 물체가 x축으로 쌓일 때 중심 간격
    COL_DY = 0.11         # 같은 y-컬럼으로 볼 횡거리 (이 안이면 x축으로 서로 막음)

    def __init__(self, objs, geom: ArenaGeometry, rng, slip_w=0.5,
                 bin_lip=False, lip_x=0.39, lip_y=0.39):
        self.objs = objs
        self.geom = geom
        self.rng = rng
        # 3D 프린트 문턱(3mm×10mm): 보관함 안쪽 두 모서리(경기장 향, 바깥 벽 제외)
        # 에만 있어 넘어 들어온 물체가 밖으로 못 나간다. lip_x/lip_y = 턱 안쪽 면.
        self.bin_lip = bin_lip
        self.lip_x = lip_x
        self.lip_y = lip_y
        self.pose = [3.8, 0.2, math.radians(90)]   # 스타트존, +y 방향
        self.payload = None
        self.front = None                           # 입구에 문 2번째 물체
        self.slip_w = slip_w                        # 이보다 빠른 회전 = 입구 물체 이탈
        self.ir = False
        self.wall_hits = 0
        self.wall_hit_at = []
        self.disturbed = 0
        self.front_slips = 0
        self.ir_forced_off_until = -1.0             # 시나리오 B 주입용

    def step(self, t, v, w):
        x, y, yaw = self.pose
        yaw2 = yaw + w * DT
        x2 = x + v * DT * math.cos((yaw + yaw2) / 2)
        y2 = y + v * DT * math.sin((yaw + yaw2) / 2)

        m = self.BODY_R
        if not (m <= x2 <= self.geom.arena_w - m and
                m <= y2 <= self.geom.arena_h - m):
            self.wall_hits += 1
            if len(self.wall_hit_at) < 5:
                self.wall_hit_at.append((round(t, 1), round(x2, 2), round(y2, 2)))
            x2 = min(self.geom.arena_w - m, max(m, x2))
            y2 = min(self.geom.arena_h - m, max(m, y2))
        self.pose = [x2, y2, wrap_angle(yaw2)]

        px = x2 + self.POCKET * math.cos(yaw2)
        py = y2 + self.POCKET * math.sin(yaw2)

        # 적재물은 빈 안에서 로봇과 함께 이동. 후진하면 바닥 마찰로 분리.
        if self.payload is not None:
            if v < -0.01:
                self._release_obj(self.payload)
                self.payload = None
            else:
                self.payload["x"], self.payload["y"] = px, py

        # 입구 물체: 구속이 없어 후진·급회전·적재물 상실 시 그 자리에 떨어진다
        if self.front is not None:
            if v < -0.01 or abs(w) > self.slip_w or self.payload is None:
                self._release_obj(self.front)
                self.front = None
                self.front_slips += 1
            else:
                self.front["x"] = x2 + self.FRONT_OFF * math.cos(yaw2)
                self.front["y"] = y2 + self.FRONT_OFF * math.sin(yaw2)

        for o in self.objs:
            if o["done"] or o is self.payload or o is self.front:
                continue
            dx, dy = o["x"] - x2, o["y"] - y2
            d = math.hypot(dx, dy)
            lat_s = -math.sin(yaw2) * dx + math.cos(yaw2) * dy   # 부호 있는 횡오차
            lat = abs(lat_s)
            ahead = math.cos(yaw2) * dx + math.sin(yaw2) * dy
            # 1차 포획: 전진 중 + 포켓 근처 + 축방향 정렬
            if self.payload is None and v > 0.01:
                if abs(ahead - self.POCKET) < 0.10 and lat < self.CAPTURE_LAT:
                    o["payload"] = True
                    self.payload = o
                    continue
            # 2차 포획: A 적재 상태로 전진, 입구 위치에 정렬 접촉
            elif self.front is None and v > 0.01:
                if abs(ahead - self.FRONT_OFF) < 0.09 and lat < self.CAPTURE_LAT:
                    o["payload"] = True
                    self.front = o
                    continue
            # 전방 스쿱 개구부: 몸체(원)로 안 밀고, 깔때기 날개가 물체를 중심축으로
            # 유도한다(포켓이 verify 블라인드 안쪽이라 블라인드 푸시의 소량 오정렬을
            # 깔때기가 흡수 — 실제 스쿱의 물리). 정렬되면 위 1차 포획이 담는다.
            if 0.0 < ahead < self.SCOOP_REACH and lat < self.SCOOP_HALF:
                if self.payload is None and v > 0.01:
                    k = 0.25                       # 스텝당 횡오차 수렴률(깔때기)
                    o["x"] += k * lat_s * math.sin(yaw2)
                    o["y"] -= k * lat_s * math.cos(yaw2)
                continue
            # 몸체 밀침 (포획 실패 시 물체가 밀려남)
            if d < self.BODY_R + self.OBJ_R:
                push = (self.BODY_R + self.OBJ_R - d) + 0.005
                o["x"] += push * dx / max(d, 1e-6)
                o["y"] += push * dy / max(d, 1e-6)
                o["x"] = min(self.geom.arena_w - 0.05, max(0.05, o["x"]))
                o["y"] = min(self.geom.arena_h - 0.05, max(0.05, o["y"]))
                self.disturbed += 1

        self.ir = self.payload is not None and t >= self.ir_forced_off_until

    # ---------------- 적재구역 혼잡/다지기 ----------------

    def _release_obj(self, o):
        """적재물을 놓을 때: 보관함 안이면 '하역'으로 파일링, 아니면 그 자리 유실."""
        o["payload"] = False
        if o["x"] < self.STORE_X and o["y"] < self.STORE_Y:
            self._pile(o)

    def _pile(self, o):
        """x축 스택 혼잡: 같은 y-컬럼에 이미 쌓인 하역물의 입구쪽(+x) 면에 막혀
        쌓인다. 뒤(−x)는 벽이 백스톱. 입구 상한(STORE_MAX=0.36)을 넘으면 스필
        (경계 밖 → score 에서 미채점). 8cm 물체가 좁은 레인으로만 들어오므로
        x축 적층이 지배적 — depth 를 벌리거나(현 방식) 막판 다지기로 해소."""
        col = [s for s in self.objs if s is not o and s.get("stored")
               and abs(s["y"] - o["y"]) < self.COL_DY]
        if self.bin_lip:
            # 턱이 개구부(안쪽 모서리)를 막아 물체가 밖(+x/+y)으로 못 넘어간다 →
            # 넘어 들어온 물체는 턱 안쪽에 걸리고, 혼잡분은 더 안쪽(−x)으로 쌓인다.
            o["x"] = min(o["x"], self.lip_x - self.OBJ_R)
            if col:
                back = min(s["x"] for s in col)         # 이미 쌓인 것 중 가장 안쪽
                o["x"] = min(o["x"], back - self.PILE_PITCH)
            o["x"] = max(o["x"], self.OBJ_R + 0.02)     # 서벽 백스톱
            o["y"] = min(o["y"], self.lip_y - self.OBJ_R)
        elif col:
            front = max(s["x"] for s in col)
            o["x"] = max(o["x"], front + self.PILE_PITCH)
        o["stored"] = True
        o["done"] = True          # 인지/충돌 루프에서 제외

    def drop_payload(self):
        if self.payload is not None:
            self.payload["payload"] = False
            self.payload = None
        if self.front is not None:
            self.front["payload"] = False
            self.front = None

    def score(self, targets):
        """종료 시 보관함 안(경계 완전 내부) 물체 채점."""
        good = bad = 0
        pts = 0.0
        for o in self.objs:
            inside = (0.04 <= o["x"] <= 0.36 and 0.04 <= o["y"] <= 0.36)
            if not inside:
                continue
            v = 20.0 if o["set"] == "set2" else 10.0
            if o["cls"] == targets.get(o["set"]):
                pts += v
                good += 1
            else:
                pts -= 2 * v
                bad += 1
        return pts, good, bad


# ------------------------------------------------------------------ 러너

def run_match(seed, targets=None, duration=180.0, hooks=None, verbose=False,
              params_override=None, world_kw=None):
    rng = random.Random(seed)
    geom = ArenaGeometry()
    objs = make_arena(rng)
    targets = targets or {"set1": rng.choice(["octa", "dodeca", "icosa"]),
                          "set2": rng.choice(FRUITS)}
    world = World(objs, geom, rng, **(world_kw or {}))
    percep = SimPerception(objs, targets, rng)
    params = dict(match_duration_s=duration)
    if params_override:
        params.update(params_override)
    mission = MissionFSM(params=params, targets=targets, geom=geom)

    t = 0.0
    mission.start(t)
    events = []
    state_time = {}                    # 상태별 체류 시간 (시간 예산 분석용)
    deposit_times = []                 # 각 하역 완료 시각
    loc_level_fn = (hooks or {}).get("loc_level", lambda t: 0)
    n = int(duration / DT) + 20
    for _ in range(n):
        if hooks and "tick" in hooks:
            hooks["tick"](t, world, mission)
        frame = percep.frame(world.pose, rng)
        ir = world.ir and t >= world.ir_forced_off_until
        v, w, dbg = mission.update(t, tuple(world.pose), frame, ir,
                                   loc_level=loc_level_fn(t))
        percep.apply_cmds(dbg["percep_cmds"])
        state_time[mission.state] = state_time.get(mission.state, 0.0) + DT
        while len(mission.deposited) > len(deposit_times):
            deposit_times.append(round(t, 1))   # 더블 하역은 같은 시각 2건
        for e in dbg["events"]:
            events.append((round(t, 1), e))
            if verbose:
                print(f"  t={t:6.1f}  {e}")
        world.step(t, v, w)
        t += DT
        if mission.state == DONE:
            break

    pts, good, bad = world.score(targets)
    stored = [o for o in world.objs if o.get("stored")]
    spilled = sum(1 for o in stored if not (0.04 <= o["x"] <= 0.36
                                            and 0.04 <= o["y"] <= 0.36))
    return dict(points=pts, good=good, bad=bad, n_stored=len(stored),
                spilled=spilled, wall_hits=world.wall_hits,
                wall_hit_at=world.wall_hit_at, disturbed=world.disturbed,
                front_slips=world.front_slips, events=events,
                end_state=mission.state,
                holding=world.payload is not None,
                deposited=mission.deposited, t_end=t,
                state_time={k: round(v, 1) for k, v in state_time.items()},
                deposit_times=deposit_times, targets=targets)


# ------------------------------------------------------------------ 시나리오

def scenario_nominal(seeds, verbose):
    ok, total_pts, results = True, 0.0, []
    for s in range(seeds):
        r = run_match(seed=s, verbose=verbose)
        results.append(r)
        total_pts += r["points"]
        fail = r["bad"] > 0 or r["wall_hits"] > 0
        ok &= not fail
        print(f"  seed {s}: {r['points']:+5.0f}점  정상하역 {r['good']} "
              f"오픽업 {r['bad']} 벽충돌 {r['wall_hits']} "
              f"밀침 {r['disturbed']:3d}  종료 {r['end_state']}"
              + ("  << FAIL" if fail else ""))
    print(f"  평균 {total_pts / seeds:+.1f}점")
    return ok


def scenario_ir_dropout(verbose):
    def tick(t, world, mission):
        # 운반 중 0.4초 IR 끊김 3회 주입 — patience(0.8s) 안이라 무시돼야 함
        if mission.state == TRANSPORT and world.payload is not None:
            for t0 in (60.0, 90.0, 120.0):
                if t0 <= t < t0 + 0.4:
                    world.ir_forced_off_until = t + DT
    r = run_match(seed=101, hooks={"tick": tick}, verbose=verbose)
    lost = any("PAYLOAD_LOST" in e for _, e in r["events"])
    ok = (not lost) and r["bad"] == 0 and r["wall_hits"] == 0 and r["good"] >= 1
    print(f"  하역 {r['good']}  오픽업 {r['bad']}  가짜 유실판정: {lost}")
    return ok


def scenario_payload_lost(verbose):
    # 단일 운반 유실 복구 경로를 검증하는 시나리오 → 더블 캐리 off 로 명시
    # (더블 캐리 중 유실은 입구 물체가 즉시 재흡입돼 IR이 안 꺼지는 별개 경로).
    dropped = {"done": False}

    def tick(t, world, mission):
        if (not dropped["done"] and mission.state == TRANSPORT
                and world.payload is not None and t > 30):
            obj = world.payload
            world.drop_payload()
            # 옆으로 빠졌다고 가정 — 직진 재포획으로 가려지지 않게 횡변위
            yaw = world.pose[2]
            obj["x"] += 0.20 * -math.sin(yaw)
            obj["y"] += 0.20 * math.cos(yaw)
            dropped["done"] = True
    r = run_match(seed=102, hooks={"tick": tick}, verbose=verbose,
                  params_override=dict(double_carry=False))
    lost = any("PAYLOAD_LOST" in e for _, e in r["events"])
    ok = dropped["done"] and lost and r["good"] >= 1 and r["bad"] == 0
    print(f"  유실 주입: {dropped['done']}  유실 감지: {lost}  "
          f"최종 하역 {r['good']}")
    return ok


def scenario_capture_miss(verbose):
    """첫 포획 시도마다 물체가 옆으로 미끄러진다(포켓 진입 방해) → 재시도 확인."""
    missed = {}

    def tick(t, world, mission):
        if mission.state == CAPTURE and world.payload is None:
            tid = mission._target["id"] if mission._target else None
            if tid is not None and not missed.get(tid):
                # 목표 물체를 횡으로 8cm 밀어 첫 푸시를 빗맞게 만든다
                for o in world.objs:
                    if o["payload"] or o["done"]:
                        continue
                    d = math.hypot(o["x"] - world.pose[0],
                                   o["y"] - world.pose[1])
                    if d < 0.5:
                        o["x"] += 0.08
                        missed[tid] = True
                        break
    r = run_match(seed=103, hooks={"tick": tick}, verbose=verbose)
    retried = any("CAPTURE_MISSED" in e for _, e in r["events"])
    ok = r["good"] >= 1 and r["bad"] == 0 and r["wall_hits"] == 0
    print(f"  재시도 발생: {retried}  하역 {r['good']}  오픽업 {r['bad']}")
    return ok


def scenario_endgame(verbose):
    # 시간 컷오프 로직 자체의 회귀 검증 — hail_mary(기본 on)는 컷오프를 의도적으로
    # 무시하는 기능이므로 여기서는 끄고 '적재 상태 방치 금지' 불변식을 지킨다.
    r = run_match(seed=104, duration=60.0, verbose=verbose,
                  params_override=dict(hail_mary=False))
    ok = (not r["holding"]) and r["bad"] == 0 and r["wall_hits"] == 0
    print(f"  종료 상태 {r['end_state']}  적재물 방치: {r['holding']}  "
          f"하역 {r['good']}")
    return ok


def scenario_hail_mary(verbose):
    """엔드게임 헤일메리(기본 on): 컷오프로 할 일이 없는 막판에 확정 타깃을
    시도한다. 종료 시 적재 상태는 감점 0 이라 허용(holding 게이트 없음) —
    안전(오픽업·벽충돌 0)과 '컷오프-only 대비 점수 하락 없음'만 게이트."""
    r0 = run_match(seed=104, duration=60.0,
                   params_override=dict(hail_mary=False))
    r = run_match(seed=104, duration=60.0, verbose=verbose)
    ok = (r["bad"] == 0 and r["wall_hits"] == 0
          and r["points"] >= r0["points"])
    print(f"  하역 {r['good']} vs 컷오프-only {r0['good']}  "
          f"점수 {r['points']:+.0f} vs {r0['points']:+.0f}  "
          f"종료 적재: {r['holding']}  오픽업 {r['bad']}")
    return ok


def _adjacent_pair_setup(world):
    """목표 과일 큐브 2개를 보관함 가는 길목에 인접 배치 (더블 캐리 유도).

    좌표는 배치 격자(0.5 간격)에서 0.3m+ 떨어진 지점 — 기존 물체와 기억
    병합(0.22m)이 일어나지 않아야 테스트가 오염되지 않는다.
    """
    apples = [o for o in world.objs if o["set"] == "set2" and o["cls"] == "apple"]
    # 투어 레인(y=1.0) 양옆 — 첫 레인 통과에서 둘 다 분류되어 페어가 성립한다
    positions = [(2.3, 0.95), (1.7, 1.05)]
    for o, (x, y) in zip(apples[:2], positions):
        o["x"], o["y"] = x, y
        # 과일면이 어느 방향에서든 보이게 (4옆면 전부 과일 — 테스트 전용)
        o["fruit_normals"] = [i * math.pi / 2 for i in range(4)]
    # 메커니즘 테스트의 결정론화: 페어 주변을 비워 관측 라벨 혼선(인접 물체
    # 병합) 배제 — 혼잡 상황의 강건성은 명목 스윕이 담당한다
    k = 0
    for o in world.objs:
        if o in apples[:2] or o["done"]:
            continue
        for x, y in positions:
            if math.hypot(o["x"] - x, o["y"] - y) < 0.8:
                o["x"], o["y"] = 0.6 + 0.45 * k, 3.45
                k += 1
                break


def scenario_double_carry(verbose):
    done = {"setup": False}

    def tick(t, world, mission):
        if not done["setup"]:
            _adjacent_pair_setup(world)
            done["setup"] = True
    r = run_match(seed=106, targets={"set2": "apple"},
                  hooks={"tick": tick}, verbose=verbose,
                  params_override=dict(double_carry=True))   # 옵션 기능 회귀 검증
    second = any("SECOND_CAPTURED" in e for _, e in r["events"])
    ok = second and r["good"] >= 2 and r["bad"] == 0 and r["wall_hits"] == 0
    print(f"  2차 포획: {second}  하역 {r['good']}  오픽업 {r['bad']}  "
          f"하역시각 {r['deposit_times']}")
    return ok


def scenario_double_slip(verbose):
    """미끄럼 임계가 낮은 세계 — B 를 자주 흘려도 감점/파손 없이 A 는 하역."""
    done = {"setup": False}

    def tick(t, world, mission):
        if not done["setup"]:
            _adjacent_pair_setup(world)
            done["setup"] = True
    r = run_match(seed=106, targets={"set2": "apple"},
                  hooks={"tick": tick}, verbose=verbose,
                  world_kw=dict(slip_w=0.15),
                  params_override=dict(double_carry=True))   # 옵션 기능 회귀 검증
    ok = r["good"] >= 1 and r["bad"] == 0 and r["wall_hits"] == 0
    print(f"  이탈 {r['front_slips']}회  하역 {r['good']}  오픽업 {r['bad']}")
    return ok


def scenario_loc_degraded(verbose):
    def loc_level(t):
        if 20 <= t < 30:
            return 2      # 완전 불량: 정지 대기
        if 30 <= t < 40:
            return 1      # 저하: 감속
        return 0
    r = run_match(seed=105, hooks={"loc_level": loc_level}, verbose=verbose)
    ok = r["good"] >= 1 and r["bad"] == 0 and r["wall_hits"] == 0
    print(f"  하역 {r['good']}  오픽업 {r['bad']}  벽충돌 {r['wall_hits']}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8, help="명목 시나리오 시드 수")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    scenarios = [
        ("A nominal", lambda: scenario_nominal(args.seeds, args.verbose)),
        ("B ir-dropout", lambda: scenario_ir_dropout(args.verbose)),
        ("C payload-lost", lambda: scenario_payload_lost(args.verbose)),
        ("D capture-miss", lambda: scenario_capture_miss(args.verbose)),
        ("E endgame", lambda: scenario_endgame(args.verbose)),
        ("E2 hail-mary", lambda: scenario_hail_mary(args.verbose)),
        ("F loc-degraded", lambda: scenario_loc_degraded(args.verbose)),
        ("G double-carry", lambda: scenario_double_carry(args.verbose)),
        ("H double-slip", lambda: scenario_double_slip(args.verbose)),
    ]
    results = []
    for name, fn in scenarios:
        print(f"[{name}]")
        ok = fn()
        results.append((name, ok))
        print(f"  -> {'PASS' if ok else 'FAIL'}\n")

    print("=" * 46)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {name:<16} {'PASS' if ok else 'FAIL'}")
    print("=" * 46)
    print("전체:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
