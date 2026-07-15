"""미션 플래너 (ROS 의존성 없음 — sim_mission.py 로 WSL 검증).

runtime/capture_fsm.py(인지 미션 FSM) 위에 얹히는 '주행' 계층. 인지가 "무엇이/어디에
보이고 지금 잡아도 되는가"를 결정하면, 여기는 "어디로 가서 무엇을 먼저 잡고 언제
포기하는가"를 결정해 (v, w) 명령을 낸다.

룰 요약 (2026-07-14 공지 기준):
  4x4m, 물체 42지점(50cm 격자) 랜덤 배치, 세트1(정다면체 16개)+세트2(과일큐브 12개)
  동시 사용. 3분, 목표 형상+목표 과일 경기 직전 공지. 세트1 10점/세트2 20점,
  오픽업은 기본점수 2배 감점(-20/-40). 보관함=좌하단 40cm(태극기 스티커),
  스타트=우하단 40cm. 물체는 스타트에서 1m 이상.

전략 원칙:
  1. 가치/시간 탐욕 선택 — 세트2(20점) 우선, 같은 값이면 가까운 것. 목표당 왕복
     예상시간을 계산해 남은 시간에 못 끝낼 목표는 시작하지 않는다.
  2. 오탐 회피 우선 — 감점(2배)이 크므로 인지의 verify 게이트 결정(veto/승인)을
     절대 우회하지 않는다. 모호하면 잡지 않고 다음 목표로 넘어간다.
  3. 보관함 주변 스티커 지오펜스 — sticker_zone 안으로 투영되는 관측은 무시.
  4. 라이다는 위치추정 전용 — 물체 회피는 카메라 관측 누적 + 스탈 감지로 한다.

입력(update): 시각, map 자세, PerceptionFrame(인지 상태/요청/조향/관측),
IR 적재 여부, 위치추정 건강도. 출력: (v, w, dbg). dbg["percep_cmds"] 에 인지로
보낼 명령(페이즈 전환·적재 통지·에피소드 리셋)이 담긴다 — 노드가 UDP 로 중계.
"""

import math
from dataclasses import dataclass, field

from nav_core import (ArenaGeometry, ControllerConfig, DiffDriveController,
                      GridPlanner, Rect, StallDetector, wrap_angle)

# ---- 미션 상태 --------------------------------------------------------------
IDLE = "IDLE"
TOUR = "TOUR"                     # 탐색 주행 (측면 search 캠으로 훑기)
GOTO = "GOTO"                     # 선택한 후보 앞 standoff 지점으로 이동
APPROACH = "APPROACH"             # verify 페이즈: 전방캠 조향으로 정렬 접근
CAPTURE = "CAPTURE"               # blind push: 방향 고정 저속 전진, IR 대기
RETREAT = "RETREAT"               # 후진 이탈 (거부/실패/스탈 후)
TRANSPORT = "TRANSPORT"           # 적재 상태로 보관함 접근 지점까지
DEPOSIT_SHED = "DEPOSIT_SHED"     # 하역 직전 제자리 회전 — 밀항 물체 털어내기
DEPOSIT_PUSH = "DEPOSIT_PUSH"     # 보관함 안으로 밀어넣기
DEPOSIT_RELEASE = "DEPOSIT_RELEASE"  # 후진+위글로 물체 분리
PARK = "PARK"                     # 종료 대기
DONE = "DONE"

# capture_fsm 의 상태/요청 문자열 (import 없이 문자열로 결합 — 인지 프로세스와
# 프로세스가 분리되어 있고, 시뮬레이션에서도 같은 문자열을 쓴다)
FSM_CAPTURE_READY = "CAPTURE_READY"
FSM_BLIND_CAPTURE = "BLIND_CAPTURE"
FSM_VERIFY_REJECTED = "VERIFY_REJECTED"
FSM_CAPTURE_MISSED = "CAPTURE_MISSED"
FSM_LOADED = "LOADED"
REQ_REAPPROACH = "REAPPROACH"
REQ_MICRO_ADJUST = "MICRO_ADJUST"
REQ_RETREAT_RESEARCH = "RETREAT_RESEARCH"
REQ_RESEARCH_NEARBY = "RESEARCH_NEARBY"


DEFAULT_PARAMS = dict(
    match_duration_s=180.0,
    # --- 속도/기하 (실측 후 조정) ---
    cruise_v=0.15, approach_v=0.10, push_v=0.06, reverse_v=0.10,
    robot_radius=0.22,          # 외접 반경 (실측)
    payload_offset=0.26,        # 로봇 중심 -> 빈에 안착한 물체 중심 [m] (실측)
    # --- 탐색 ---
    # 배치 격자(50cm)의 행 사이 중간선을 달린다 — 몸체가 물체와 안 닿고
    # 측면캠이 양옆 행을 1.5m 안에서 훑는다. 격자 원점이 다르면 실측 후 조정.
    tour_lanes_y=(1.0, 3.0),
    tour_margin_x=0.7,          # 레인 좌우 끝 여유
    # --- 접근/포획 ---
    standoff_dist=0.60,         # 물체 앞 정지 지점 (verify 시작 거리)
    approach_stop_short=0.35,   # READY 없이는 물체 앞 이 거리까지만 전진.
                                # 포켓(payload_offset 0.26)보다 충분히 길어야
                                # 위치 오차에도 승인 없는 스쿱이 안 생긴다.
    approach_range_guard=0.32,  # verify 관측 거리가 이보다 가까운데 READY 가
                                # 없으면 즉시 중단 (이동거리 기준의 이중 안전)
    approach_timeout_s=14.0,
    capture_push_max=0.45,      # blind push 최대 전진 거리
    capture_push_limit_s=6.0,   # configs/*.yaml verify.capture_push_limit 와 일치
    retreat_dist=0.30,
    micro_adjust_dist=0.15, micro_adjust_yaw_deg=35.0,  # 과일면이 다른 45° 섹터에
                                                        # 있을 수 있어 크게 돈다
    max_tries_per_object=2,     # 접근 실패 허용 횟수 (초과 시 블랙리스트)
    ir_lost_patience_s=0.8,     # 운반 중 IR 순간 끊김 허용
    # --- 2개 운반 (더블 캐리): 빈 안쪽 1개(IR 확인) + 입구에 1개 더 물고 운반 ---
    double_carry=True,          # A 포획 후 근처 확정 타깃 B를 입구에 물고 같이 하역
    pair_max_dist=1.5,          # A 포획 지점에서 B 후보 최대 거리
    pair_max_turn_deg=70.0,     # B에서 보관함 방향으로 꺾이는 각 상한 (커브 완만해야
                                # 입구의 B가 안 빠짐)
    second_push_dist=0.22,      # B 확보 푸시 거리 — 입구 물체는 IR 확인 불가(개루프)
    carry_w_max=0.5,            # B 운반 중 각속도 상한 [rad/s] (원심 이탈 방지)
    carry_v_max=0.12,           # B 운반 중 속도 상한
    deposit_depth_double=0.17,  # 2개 하역 시 A(포켓) 목표 x — B는 그 앞 ~0.09
    t_capture2_est=6.0,         # B 확보 예상 시간 (시간 예산용)
    # 밀항 물체 대응: 운반 중 입구에 스스로 붙은 정체불명 물체가 보관함까지
    # 따라 들어가면 조용히 -40. 센서로 감지 불가 → 하역 직전 제자리 고속 회전
    # (미끄럼 임계 초과)으로 털어낸다. 의도한 B 를 물었을 때는 생략.
    shed_spin=True,
    shed_spin_angle_deg=330.0,
    shed_spin_w=1.1,            # carry_w_max/미끄럼 임계보다 확실히 크게
    # --- 하역 ---
    deposit_approach_x=0.95,    # 보관함 진입 직전 정렬 지점 x (진행방향 -x)
    deposit_lane_y=0.30,        # 진입 레인 y — 도착오차+요오차로 벽쪽 드리프트해도
                                # 남벽과 안 닿고, 물체(반폭 4cm)는 경계 안(0.36)에 듦
    deposit_depths=(0.12, 0.22, 0.31),  # 물체 중심 목표 x — 하역마다 순환, 겹치면
                                        # 서쪽 벽 쪽으로 밀려 쌓임 (벽이 백스톱)
    deposit_push_v=0.10,        # 하역 진입 속도 — 검증 끝난 구간이라 포획 푸시보다
                                # 빨라도 됨 (트립당 고정 오버헤드 절감)
    release_reverse_dist=0.40,
    release_reverse_v=0.12,
    release_wiggle_w=0.5, release_wiggle_period_s=0.6,
    # --- 목표 선택 / 시간 예산 ---
    value_set1=10.0, value_set2=20.0,
    value_unknown=4.0,          # 미확인 후보를 조사하러 갈 기대 가치
    unknown_gate="full_tour",   # 미확인 조사 허용 시점: full_tour(투어 1바퀴 후) |
                                # first_lane(첫 레인 후) | always(즉시)
    eff_speed=0.11,             # 왕복 예상시간 계산용 유효 속도 (회전 포함)
    t_approach_est=10.0, t_capture_est=6.0, t_deposit_est=11.0,
    endgame_margin_s=8.0,       # 이 여유가 없으면 새 목표를 시작하지 않음
    # --- 위치추정 건강도 대응 ---
    loc_degraded_scale=0.4,     # 연속 거부 5~12회: 감속
    # --- 조향 ---
    steering_gain=1.2, steering_sign=-1.0,  # 오프셋 +(우측) -> w 음수(우회전)
    memory_merge_dist=0.22,     # 이 거리 안 관측은 같은 물체로 병합
                                # (배치 간격 50cm 의 절반 미만이어야 안 섞임)
    memory_max_range=2.2,       # 이보다 먼 관측은 위치 오차가 커서 기억 안 함
)


# ------------------------------------------------------------- 인지 입력 프레임

@dataclass
class PerceptionFrame:
    """노드(UDP)/시뮬레이션이 채워 주는 인지 요약. 없으면 필드 None/[] 유지."""
    fsm_state: str = "SEARCHING"
    request: str = None
    steering: dict = None           # {combined_offset_px, allowed_offset_px, aligned}
    verify_range: float = None      # verify 캠이 지금 판정 중인 물체의 거리 추정 [m]
                                    # (veto 를 의도한 목표에 귀속시켜도 되는지 검증용)
    verify_bearing: float = None    # 그 물체의 로봇 기준 방위 [rad] — 없이 거리만
                                    # 쓰면 시야각 가장자리 물체의 위치가 크게 틀린다
    # sightings: [{set:"set1"|"set2", cls:str|None, state:str, bearing:rad, range:m}]
    sightings: list = field(default_factory=list)


# ------------------------------------------------------------------ 물체 기억

class ObjectMemory:
    """카메라 관측 누적 → map 좌표 물체 목록 (경로계획 장애물 + 목표 후보).

    cube_hunt(세트1 목표가 cube 로 공지된 경기)에서는 'cube' 무지면 관측의
    방위 섹터를 물체별로 누적한다. 옆면 4개(4사분면)가 전부 무지로 관측되고
    과일 관측이 한 번도 없으면 세트1 큐브로 확정 — 세트2는 과일면이 3개라
    옆면 4개가 모두 무지일 수 없다(윗면+밑면 2자리뿐이라 모순). 단일 시점
    분류로는 원리적으로 불가능한 구분이 위치 기반 다시점 누적으로 가능해진다.
    """

    # 인지 search 상태 → 확신 랭크. 3 = 분류 확정(타깃 여부는 cls 로 판정),
    # 1~2 = 조사 가치 있는 후보, 0 = 스침. (set1/set2 정책 상태 이름 모두 수용)
    _RANK = {"SEARCHING": 0, "GIVE_UP": 0,
             "FAR_CANDIDATE": 1, "UNKNOWN_CUBE": 1, "CUBE_BLANK_VIEW": 1,
             "TARGET_CANDIDATE": 2,
             "TARGET_CONFIRMED": 3, "NON_TARGET": 3, "NON_TARGET_FRUIT": 3,
             "REJECTED": 3}

    def __init__(self, merge_dist, max_range=2.2, cube_hunt=False):
        self.merge_dist = merge_dist
        self.max_range = max_range
        self.cube_hunt = cube_hunt
        self.objects = []   # dict: x,y,set,cls,rank,status,tries,last_seen
        self._next_id = 0

    def integrate(self, t, pose, sightings, geom: ArenaGeometry):
        from nav_core import project_to_map
        for s in sightings:
            if s["range"] > self.max_range:   # 원거리 단안 거리는 오차가 크다
                continue
            x, y = project_to_map(pose, s["bearing"], s["range"])
            if not geom.in_arena(x, y, 0.05):
                continue
            if geom.sticker_zone.contains(x, y):   # 태극기 지오펜스
                continue
            rank = self._RANK.get(s.get("state"), 0)
            ent = self._nearest(x, y)
            if ent is None:
                ent = dict(id=self._next_id, x=x, y=y, set=None, cls=None,
                           rank=0, status="open", tries=0, last_seen=t,
                           votes={}, blank_sectors=set())
                self.objects.append(ent)
                self._next_id += 1
            else:
                a = 0.3   # 위치는 지수평활 (단안 거리 노이즈 완화)
                ent["x"] = (1 - a) * ent["x"] + a * x
                ent["y"] = (1 - a) * ent["y"] + a * y
                ent["last_seen"] = t
            ent["rank"] = max(ent["rank"], rank)
            if s.get("cls") == "cube":
                # 무지면 큐브 관측: 라벨 투표가 아니라 관측 방위 섹터(16분할)를
                # 누적한다 — 확정 규칙은 _maybe_confirm_cube 참조.
                view = math.atan2(pose[1] - ent["y"], pose[0] - ent["x"])
                sec = int((view + math.pi) / (2 * math.pi) * 16) % 16
                ent["blank_sectors"].add(sec)
                self._maybe_confirm_cube(ent)
            elif s.get("cls"):
                # 라벨은 투표 + 히스테리시스: 인접 물체 관측이 병합 반경에
                # 섞여도 확정 라벨이 한두 프레임에 뒤집히지 않게 한다.
                key = (s.get("set"), s["cls"])
                ent["votes"][key] = ent["votes"].get(key, 0) + 1
                cur = (ent.get("set"), ent.get("cls"))
                best = max(ent["votes"], key=ent["votes"].get)
                if best != cur:
                    lead = ent["votes"][best] - ent["votes"].get(cur, 0)
                    if ent.get("cls") is None or lead >= 2:
                        ent["set"], ent["cls"] = best

    def _maybe_confirm_cube(self, ent):
        """무지면 뷰 각도 커버리지 + 과일 관측 전무 → 세트1 큐브 확정.

        건전성: 무지 뷰는 시선 ±45° 내 법선의 옆면을 무지로 인증한다(분류기
        가시 한계 ~65° > 45°). 옆면 법선 방향은 미지이므로 '모든 방향이 어떤
        무지 뷰의 ±45° 안'이어야 한다 = 연속 관측 각도 갭 < 90°. 16섹터
        양자화 보수 조건: 이웃 점유 섹터 사이 빈 섹터 ≤ 1 (갭 ≤ 67.5°).
        사분면 집합 검사로는 경계에 몰린 뷰들이 88° 갭을 남길 수 있어 부족.
        """
        if not self.cube_hunt or ent["votes"]:
            return
        # 새 룰(반대편 2면): set2 큐브의 무지 사각은 항상 180° 떨어진 두 50°
        # 창뿐 → 그런 큐브의 blank 관측은 최소 130° 연속 갭을 남긴다. 따라서
        # '모든 갭 ≤ 90°' 를 요구하면 set2 는 원리적으로 통과 불가(오인증 0).
        # 단일 시점 분류 노이즈 방어로 최소 4섹터 + 갭≤90° 를 보수적으로 유지.
        occ = sorted(ent["blank_sectors"])
        if len(occ) < 4:
            return
        for i, s in enumerate(occ):
            nxt = occ[(i + 1) % len(occ)]
            gap = (nxt - s) % 16
            if gap > 4:       # 4섹터 = 90°. 초과 = set2 반대편-2면 가설 배제 실패
                return
        ent["set"], ent["cls"] = "set1", "cube"
        ent["rank"] = max(ent["rank"], 3)
        if ent["status"] == "defer":
            ent["status"] = "open"    # 인증 완료 — 다시 포획 후보

    def _nearest(self, x, y):
        best, bd = None, self.merge_dist
        for o in self.objects:
            d = math.hypot(o["x"] - x, o["y"] - y)
            if d < bd:
                best, bd = o, d
        return best

    def obstacles(self, exclude_id=None):
        # captured = 지금 빈 안에 있음 → 기억 속 위치는 낡았으니 장애물 아님
        return [(o["x"], o["y"]) for o in self.objects
                if o["status"] not in ("deposited", "captured")
                and o["id"] != exclude_id]

    def add_virtual(self, x, y):
        """스탈 감지 시 코앞에 임시 장애물 등록 (안 보였던 물체)."""
        self.objects.append(dict(id=self._next_id, x=x, y=y, set=None, cls=None,
                                 rank=0, status="virtual", tries=0,
                                 last_seen=0.0, votes={}, blank_sectors=set()))
        self._next_id += 1


# ------------------------------------------------------------------ 미션 FSM

class MissionFSM:
    def __init__(self, params=None, targets=None, geom=None):
        self.p = dict(DEFAULT_PARAMS)
        if params:
            self.p.update(params)
        self.targets = targets or {}          # {"set1": "icosa", "set2": "apple"}
        self.geom = geom or ArenaGeometry()
        ctrl_cfg = ControllerConfig(max_v=self.p["cruise_v"])
        self.ctrl = DiffDriveController(ctrl_cfg)
        self.planner = GridPlanner(self.geom, robot_radius=self.p["robot_radius"])
        self.stall = StallDetector()
        # cube 공지 경기: 다시점 무지면 증명이 필요 — 외곽 일주 투어 + 섹터 누적
        self.cube_hunt = (self.targets or {}).get("set1") == "cube"
        self.memory = ObjectMemory(self.p["memory_merge_dist"],
                                   self.p["memory_max_range"],
                                   cube_hunt=self.cube_hunt)

        self.state = IDLE
        self.t_start = None
        self.deposited = []                    # [(set, cls)] 하역 완료 기록
        self.score = 0.0
        self._tour_idx = 0
        self._tour_wps = self._build_tour()
        self._tour_route = []
        self._tour_pass_done = False           # 1바퀴 끝나야 미확인 조사 허용
        self._unintended_ir_since = None
        self._last_time_check = -1.0
        self._recovering = False    # 운반 중 유실물 회수 모드 (재검증 생략)
        self._shed = None
        self._vr_hist = []          # verify_range 최근 샘플 (중앙값 필터용)
        self._route = []
        self._target = None                    # 현재 접근/포획 중인 memory 객체
        self._payload_obj = None               # 빈 안쪽에 안착(IR 확인)된 객체
        self._front_obj = None                 # 입구에 물고 가는 2번째 객체 (무확인)
        self._retreat = None                   # dict(start_xy, dist, then)
        self._push = None                      # dict(start_xy, yaw_lock, t0)
        self._release = None
        self._ir_lost_since = None
        self._approach_t0 = None
        self._approach_start = None
        self._last_t = None
        self._cmds = []
        self._events = []
        self._prev_v = 0.0

    # ---------------- 외부 API ----------------

    def start(self, t):
        self.t_start = t
        self._set_state(TOUR, t)

    def remaining(self, t):
        if self.t_start is None:
            return self.p["match_duration_s"]
        return self.p["match_duration_s"] - (t - self.t_start)

    def update(self, t, pose, percep: PerceptionFrame, ir_loaded, loc_level=0):
        """loc_level: 0 정상 / 1 저하(감속) / 2 불량(정지 대기)."""
        dt = 0.05 if self._last_t is None else max(1e-3, t - self._last_t)
        self._last_t = t
        self._cmds, self._events = [], []

        if self.state == IDLE or self.state == DONE:
            return 0.0, 0.0, self._dbg()

        if self.remaining(t) <= 0.0:
            self._set_state(DONE, t)
            return 0.0, 0.0, self._dbg()

        # 관측 누적 (운반/하역 중엔 새 후보에 관심 없지만 장애물 지도는 계속 갱신)
        self.memory.integrate(t, pose, percep.sightings, self.geom)

        # 위치추정 불량: 이동 상태에서는 멈춰서 재수렴 대기 (포획 푸시는 계속 —
        # 그 구간은 어차피 odom 방향 유지 + IR 이 지배)
        if loc_level >= 2 and self.state in (TOUR, GOTO, TRANSPORT):
            v, w = self.ctrl._limit(0.0, 0.0, dt)
            return v, w, self._dbg()
        speed_scale = self.p["loc_degraded_scale"] if loc_level == 1 else 1.0

        # 의도치 않은 포획: 포획 단계가 아닌데 IR이 켜짐 = 정체불명 물체가 빈에
        # 들어옴. 무엇인지 모르니 즉시 후진 배출 (오픽업 감점 원천 차단).
        # 예외 1: 유실물 회수 중이면 그 물체는 이미 검증된 타깃 — 재적재로 처리.
        # 예외 2: 더블 캐리로 A 를 적재한 채 B 로 가는 중이면 IR 켜짐이 정상.
        if ir_loaded and self._payload_obj is None and \
                self.state in (TOUR, GOTO, APPROACH, PARK):
            if self._recovering and self._target is not None:
                self._events.append("RECAPTURED_LOST_PAYLOAD")
                self._cmds.append(dict(cmd="note_loaded", loaded=True))
                self._target["status"] = "captured"
                self._payload_obj = self._target
                self._target = None
                self._recovering = False
                self._unintended_ir_since = None
                self._plan_to_deposit(pose)
                self._set_state(TRANSPORT, t)
                v, w = self.ctrl._limit(0.0, 0.0, dt)
                return v, w, self._dbg()
            if self._unintended_ir_since is None:
                self._unintended_ir_since = t
            elif t - self._unintended_ir_since > 0.4:
                self._events.append("UNINTENDED_CAPTURE->DROP")
                if self._target is not None and self._target["status"] == "active":
                    self._target["status"] = "open"
                self._target = None
                self._unintended_ir_since = None
                # 배출 지점(포켓 위치)을 장애물로 등록 — 같은 물체 재포획 루프 방지
                self.memory.add_virtual(
                    pose[0] + self.p["payload_offset"] * math.cos(pose[2]),
                    pose[1] + self.p["payload_offset"] * math.sin(pose[2]))
                self._begin_retreat(pose, 0.35, then=TOUR)
                # 배출 물체·가상 장애물이 쌓인 구석에서는 경로가 전부 막혀
                # 직선 폴백이 재스쿱을 유발한다 — 중심 쪽으로 강제 이탈 후 재개
                cx = self.geom.arena_w / 2.0
                cy = self.geom.arena_h / 2.0
                dn = math.hypot(cx - pose[0], cy - pose[1])
                if dn > 0.1:
                    self._retreat["escape_wp"] = (
                        pose[0] + 0.9 * (cx - pose[0]) / dn,
                        pose[1] + 0.9 * (cy - pose[1]) / dn)
        else:
            self._unintended_ir_since = None

        # 진행 중 시간 재검사: 지금 위치 기준으로 남은 여정을 끝낼 수 없으면 포기.
        # A 적재 중(B 사냥)이면 B 만 포기하고 하역은 반드시 간다.
        if (self.state in (GOTO, APPROACH) and self._target is not None
                and t - self._last_time_check > 1.0):
            self._last_time_check = t
            tgt = self._target
            d1 = math.hypot(tgt["x"] - pose[0], tgt["y"] - pose[1])
            d2 = math.hypot(tgt["x"] - self.p["deposit_approach_x"],
                            tgt["y"] - self.p["deposit_lane_y"])
            need = (d1 + d2) / self.p["eff_speed"] + \
                self.p["t_capture_est"] + self.p["t_deposit_est"]
            if need > self.remaining(t) - 2.0:
                self._events.append("TIME_ABORT")
                tgt["status"] = "open"
                self._target = None
                if self._payload_obj is not None:
                    self._plan_to_deposit(pose)
                    self._set_state(TRANSPORT, t)
                else:
                    self._set_state(TOUR, t)

        handler = {
            TOUR: self._st_tour, GOTO: self._st_goto,
            APPROACH: self._st_approach, CAPTURE: self._st_capture,
            RETREAT: self._st_retreat, TRANSPORT: self._st_transport,
            DEPOSIT_SHED: self._st_deposit_shed,
            DEPOSIT_PUSH: self._st_deposit_push,
            DEPOSIT_RELEASE: self._st_deposit_release,
            PARK: self._st_park,
        }[self.state]
        v, w = handler(t, dt, pose, percep, ir_loaded)
        v *= speed_scale
        # 입구에 B 를 물고 있는 동안은 완만하게 (급회전 = B 이탈)
        if self._front_obj is not None and v > 0:
            v = min(v, self.p["carry_v_max"])
            w = max(-self.p["carry_w_max"], min(self.p["carry_w_max"], w))

        # 스탈 감지 → 임시 장애물 등록. 적재 중엔 후진 금지(빈이 앞이 열려 있어
        # 후진하면 적재물이 빠진다) — 제자리 회전 재계획으로 우회한다.
        if self.state in (TOUR, GOTO, TRANSPORT) and \
                self.stall.update(dt, v, pose):
            nose = (pose[0] + 0.3 * math.cos(pose[2]),
                    pose[1] + 0.3 * math.sin(pose[2]))
            self.memory.add_virtual(*nose)
            if self._payload_obj is not None:
                self._events.append("STALL->REPLAN(payload)")
                if self.state == TRANSPORT:
                    self._plan_to_deposit(pose)
                elif self.state == GOTO and self._target is not None:
                    self._plan_to_standoff(pose, self._target)
            else:
                self._events.append("STALL->RETREAT")
                self._begin_retreat(pose, self.p["retreat_dist"],
                                    then=self.state)
        self._prev_v = v
        return v, w, self._dbg()

    # ---------------- 상태 구현 ----------------

    def _st_tour(self, t, dt, pose, percep, ir):
        target = self._select_target(t, pose)
        if target is not None:
            self._target = target
            target["status"] = "active"
            self._plan_to_standoff(pose, target)
            self._set_state(GOTO, t)
            return self.ctrl._limit(self._prev_v, 0.0, dt)

        if self.remaining(t) < self.p["endgame_margin_s"]:
            self._set_state(PARK, t)
            return self.ctrl._limit(0.0, 0.0, dt)

        if not self._tour_route:
            wp = self._tour_wps[self._tour_idx % len(self._tour_wps)]
            route = self.planner.plan((pose[0], pose[1]), wp,
                                      self.memory.obstacles(),
                                      [self.geom.storage])
            self._tour_route = route if route else [wp]
        v, w, done = self.ctrl.go_to(pose, self._tour_route[0], dt)
        if done:
            self._tour_route.pop(0)
            if not self._tour_route:
                self._tour_idx += 1
                if self._tour_idx >= len(self._tour_wps):
                    self._tour_pass_done = True
        return v, w

    def _visit_complete(self, t):
        """cube 보완 방문: 목표 갭 섹터가 채워졌으면 즉시 종료 (READY 불필요 —
        방문 목적은 관측이고, search 캠이 1.5m 에서 이미 기록했을 수 있다)."""
        tgt = self._target
        if not (self.cube_hunt and tgt is not None
                and not self._is_definite_target(tgt)
                and tgt.get("visit_sec") is not None):
            return False
        vs = tgt["visit_sec"]
        if any((vs + k) % 16 in tgt["blank_sectors"] for k in (-1, 0, 1)):
            self._events.append("SECTOR_VISIT_DONE")
            tgt["status"] = "defer"
            tgt["visits"] = tgt.get("visits", 0) + 1
            tgt["visit_sec"] = None
            self._target = None
            self._set_state(TOUR, t)
            return True
        return False

    def _st_goto(self, t, dt, pose, percep, ir):
        if self._visit_complete(t):
            return self.ctrl._limit(self._prev_v, 0.0, dt)
        if not self._route:
            self._enter_approach(t, pose)
            return self.ctrl._limit(self._prev_v, 0.0, dt)
        wp = self._route[0]
        last = len(self._route) == 1
        v, w, done = self.ctrl.go_to(pose, wp, dt,
                                     final_yaw=self._face_target(pose) if last else None)
        if done:
            self._route.pop(0)
            if not self._route:
                self._enter_approach(t, pose)
        return v, w

    def _abandon_second(self, t, pose, reason, blacklist=False):
        """B 사냥 포기 → A 하역으로 전환 (적재 중엔 후진/재접근 안 함)."""
        if self._target is not None:
            if blacklist:
                self._target["status"] = "blacklist"
            elif self._target["status"] == "active":
                self._target["status"] = "open"
        self._events.append(f"SECOND_ABANDONED({reason})")
        self._target = None
        self._plan_to_deposit(pose)
        self._set_state(TRANSPORT, t)

    def _st_approach(self, t, dt, pose, percep, ir):
        p = self.p
        if self._visit_complete(t):
            return self.ctrl._limit(0.0, 0.0, dt)
        armed = t - self._approach_t0 > 0.3   # 진입 직후 이전 에피소드 판정 무시
        hunting_second = self._payload_obj is not None

        # 인지 요청/판정 우선 처리
        if armed and percep.fsm_state == FSM_VERIFY_REJECTED:
            # veto 는 '시야의 가장 가까운 물체'에 대한 판정 — 내 목표가 맞는지
            # 거리로 귀속 확인. 다른 물체였으면 그놈만 기억에서 제외하고 재접근.
            vetoed_other = False
            if percep.verify_range is not None and self._target is not None:
                ang = pose[2] + (percep.verify_bearing or 0.0)
                vx = pose[0] + percep.verify_range * math.cos(ang)
                vy = pose[1] + percep.verify_range * math.sin(ang)
                if math.hypot(vx - self._target["x"],
                              vy - self._target["y"]) > 0.25:
                    vetoed_other = True
                    other = self.memory._nearest(vx, vy)
                    if other is not None and other is not self._target:
                        other["status"] = "blacklist"
                    else:
                        self.memory.add_virtual(vx, vy)
            if hunting_second:
                # 적재 중엔 후진 재접근 불가 — B 만 포기하고 하역 간다
                self._abandon_second(t, pose, "veto",
                                     blacklist=not vetoed_other)
            elif vetoed_other and self._target["tries"] < p["max_tries_per_object"]:
                self._target["tries"] += 1
                self._events.append("VETO_OTHER->REAPPROACH")
                self._begin_retreat(pose, p["retreat_dist"], then=GOTO)
            else:
                self._blacklist(self._target, "veto")
                self._events.append("VETO->RETREAT")
                self._begin_retreat(pose, p["retreat_dist"], then=TOUR)
            return self.ctrl._limit(0.0, 0.0, dt)
        # 유실물 회수: 분류는 이미 끝난 물체 — 조향 정렬만 되면 바로 push.
        # unknown(과일면 안 보임)으로 인한 MICRO_ADJUST 도 무시한다.
        if self._recovering:
            if (percep.steering and percep.steering.get("aligned") and
                    percep.verify_range is not None and
                    percep.verify_range < 0.5):
                self._push = dict(start=(pose[0], pose[1]), yaw_lock=pose[2],
                                  t0=t)
                self._set_state(CAPTURE, t)
                return self.ctrl.straight(p["push_v"], dt, hold_yaw_err=0.0)
        elif armed and percep.request == REQ_MICRO_ADJUST:
            if hunting_second:
                self._abandon_second(t, pose, "unknown")
                return self.ctrl._limit(0.0, 0.0, dt)
            self._target["tries"] += 1
            if self._target["tries"] > p["max_tries_per_object"]:
                self._blacklist(self._target, "unknown_persist")
                self._begin_retreat(pose, p["retreat_dist"], then=TOUR)
            else:
                self._events.append("MICRO_ADJUST")
                self._begin_retreat(pose, p["micro_adjust_dist"], then=GOTO,
                                    yaw_jitter=math.radians(
                                        p["micro_adjust_yaw_deg"]) *
                                    (1 if self._target["tries"] % 2 else -1))
            return self.ctrl._limit(0.0, 0.0, dt)
        if armed and percep.fsm_state in (FSM_CAPTURE_READY, FSM_BLIND_CAPTURE):
            # cube 공지 경기의 READY 는 '무지면 큐브'라는 뜻일 뿐이다 (실물
            # set1 분류기도 무지면 세트2 큐브를 'cube'로 본다). 포획 허가는
            # 내비게이터의 다시점 인증이 최종 결정: 미인증 항목이면 READY 무시
            # — 이 접근은 조사일 뿐이고, 무지면만 봤다면 섹터 증거로 충분하다.
            if (self.cube_hunt and self._target is not None
                    and not self._is_definite_target(self._target)):
                # 조사 결과 '무지면 큐브'만 확인 — 이 시점 관측은 이미 섹터에
                # 반영됐다. 재접근해도 같은 각도라 증거가 안 늘므로 보류(defer)
                # 처리하고 투어를 계속한다. 섹터가 채워져 인증되면
                # _maybe_confirm_cube 가 open 으로 복귀시킨다.
                self._events.append("READY_UNCERTIFIED->DEFER")
                self._target["status"] = "defer"
                self._target["visits"] = self._target.get("visits", 0) + 1
                if hunting_second:
                    self._abandon_second(t, pose, "uncertified")
                else:
                    self._begin_retreat(pose, p["retreat_dist"], then=TOUR)
                return self.ctrl._limit(0.0, 0.0, dt)
            # verify 가 잠근 물체가 바로 그 인증된 항목인지 위치로 확인 —
            # 옆의 미인증(세트2일 수도) 큐브면 재접근.
            if (self.cube_hunt and percep.verify_range is not None
                    and self._target is not None):
                ang = pose[2] + (percep.verify_bearing or 0.0)
                vx = pose[0] + percep.verify_range * math.cos(ang)
                vy = pose[1] + percep.verify_range * math.sin(ang)
                if math.hypot(vx - self._target["x"],
                              vy - self._target["y"]) > 0.25:
                    self._events.append("READY_OTHER_CUBE->REAPPROACH")
                    self._target["tries"] += 1
                    if self._target["tries"] > p["max_tries_per_object"]:
                        self._blacklist(self._target, "ready_other")
                        nxt = TOUR
                    else:
                        nxt = GOTO
                    if hunting_second:
                        self._abandon_second(t, pose, "ready_other")
                    else:
                        self._begin_retreat(pose, p["retreat_dist"], then=nxt)
                    return self.ctrl._limit(0.0, 0.0, dt)
            self._push = dict(start=(pose[0], pose[1]), yaw_lock=pose[2], t0=t)
            self._set_state(CAPTURE, t)
            return self.ctrl.straight(p["push_v"], dt, hold_yaw_err=0.0)

        # verify 관측으로 목표 위치 갱신되므로 물체를 계속 바라보며 전진
        tgt = self._target
        dist = math.hypot(tgt["x"] - pose[0], tgt["y"] - pose[1])
        herr = wrap_angle(math.atan2(tgt["y"] - pose[1], tgt["x"] - pose[0])
                          - pose[2])
        w_des = 2.0 * herr
        if percep.steering and percep.steering.get("allowed_offset_px"):
            off = (percep.steering["combined_offset_px"] /
                   max(1.0, percep.steering["allowed_offset_px"]))
            off = max(-1.5, min(1.5, off))
            w_des = p["steering_sign"] * p["steering_gain"] * off

        traveled = math.hypot(pose[0] - self._approach_start[0],
                              pose[1] - self._approach_start[1])
        max_travel = max(0.15, p["standoff_dist"] - p["approach_stop_short"])
        # 관측 거리 가드: READY 없이 물체가 너무 가까우면 (위치 추정 오차로
        # 이동거리 상한이 못 막은 경우) 즉시 중단 — 승인 없는 스쿱 방지.
        # 단안 거리는 노이즈가 커서 최근 3샘플 중앙값으로 판정한다.
        if percep.verify_range is not None:
            self._vr_hist.append(percep.verify_range)
            if len(self._vr_hist) > 3:
                self._vr_hist.pop(0)
        too_close = (len(self._vr_hist) == 3 and
                     sorted(self._vr_hist)[1] < p["approach_range_guard"] and
                     not self._recovering)
        if (t - self._approach_t0 > p["approach_timeout_s"] or
                traveled > max_travel or too_close):
            if hunting_second:
                # 적재 중이라 후진 재접근은 불가 — 전진 경로로 standoff 재계획
                self._target["tries"] += 1
                if self._target["tries"] <= p["max_tries_per_object"]:
                    self._events.append("REAPPROACH_SECOND")
                    self._plan_to_standoff(pose, self._target)
                    self._set_state(GOTO, t)
                else:
                    self._abandon_second(t, pose, "approach_timeout")
                return self.ctrl._limit(0.0, 0.0, dt)
            self._target["tries"] += 1
            if self._target["tries"] > p["max_tries_per_object"]:
                self._blacklist(self._target, "approach_timeout")
                nxt = TOUR
            else:
                nxt = GOTO
            self._events.append("APPROACH_TIMEOUT")
            self._begin_retreat(pose, p["retreat_dist"], then=nxt)
            return self.ctrl._limit(0.0, 0.0, dt)

        v_des = p["approach_v"] if dist > 0.15 else p["push_v"]
        return self.ctrl._limit(v_des, max(-0.8, min(0.8, w_des)), dt)

    def _st_capture(self, t, dt, pose, percep, ir):
        p = self.p
        if self._payload_obj is not None:      # A 적재 상태 → 2차 포획 로직
            return self._st_capture_second(t, dt, pose, percep, ir)
        pushed_now = math.hypot(pose[0] - self._push["start"][0],
                                pose[1] - self._push["start"][1])
        # 푸시가 거의 시작되기도 전에 IR 이 켜져 있으면, 승인 직전에 다른 물체가
        # 먼저 빈에 들어와 있던 것 (READY 는 그 뒤의 진짜 타깃을 보고 떴을 수
        # 있다). 정상 안착은 물리적으로 10cm+ 푸시가 필요 — 즉시 배출한다.
        if ir and pushed_now < 0.05 and t - self._push["t0"] < 0.6:
            self._events.append("PREOCCUPIED_BIN->DROP")
            if self._target is not None and self._target["status"] == "active":
                self._target["status"] = "open"
            self._target = None
            self._recovering = False
            self.memory.add_virtual(
                pose[0] + p["payload_offset"] * math.cos(pose[2]),
                pose[1] + p["payload_offset"] * math.sin(pose[2]))
            self._begin_retreat(pose, 0.35, then=TOUR)
            return self.ctrl._limit(0.0, 0.0, dt)
        if ir:   # 안착 성공 (1차 포획)
            self._cmds.append(dict(cmd="note_loaded", loaded=True))
            self._target["status"] = "captured"
            self._payload_obj = self._target
            self._target = None
            self._recovering = False
            self._events.append("LOADED")
            self._ir_lost_since = None
            # 더블 캐리: 근처의 확정 타깃을 입구에 물고 같이 하역
            b = self._select_second(t, pose)
            if b is not None:
                b["status"] = "active"
                self._target = b
                self._events.append(f"SECOND_TARGET({b['id']})")
                self._plan_to_standoff(pose, b)
                self._set_state(GOTO, t)
            else:
                self._plan_to_deposit(pose)
                self._set_state(TRANSPORT, t)
            return self.ctrl._limit(0.0, 0.0, dt)

        pushed = math.hypot(pose[0] - self._push["start"][0],
                            pose[1] - self._push["start"][1])
        if (percep.fsm_state == FSM_CAPTURE_MISSED or
                pushed > p["capture_push_max"] or
                t - self._push["t0"] > p["capture_push_limit_s"]):
            self._target["tries"] += 1
            self._events.append("CAPTURE_MISSED")
            if self._target["tries"] > p["max_tries_per_object"]:
                self._blacklist(self._target, "capture_missed")
                nxt = TOUR
            else:
                nxt = GOTO
            self._begin_retreat(pose, p["retreat_dist"] + pushed, then=nxt)
            return self.ctrl._limit(0.0, 0.0, dt)

        herr = wrap_angle(self._push["yaw_lock"] - pose[2])
        return self.ctrl.straight(p["push_v"], dt, hold_yaw_err=herr)

    def _st_capture_second(self, t, dt, pose, percep, ir):
        """2차 포획: A(빈 안, IR 확인됨)를 실은 채 B 를 입구로 밀어 확보.

        입구 물체는 센서 확인이 불가하므로 성공 판정은 개루프(푸시 거리 완료).
        IR 이 꺼지면 푸시 반동으로 A 가 빠진 것 — 즉시 A 회수 플로우로 전환.
        """
        p = self.p
        if not ir:   # A 이탈
            self._events.append("PAYLOAD_LOST(during_second)")
            self._cmds.append(dict(cmd="note_payload_lost"))
            a = self._payload_obj
            a["x"] = pose[0] + p["payload_offset"] * math.cos(pose[2])
            a["y"] = pose[1] + p["payload_offset"] * math.sin(pose[2])
            a["status"] = "open"
            if self._target is not None:
                self._target["status"] = "open"
            self._payload_obj = None
            self._target = a
            self._recovering = True
            self._begin_retreat(pose, p["retreat_dist"], then=GOTO)
            return self.ctrl._limit(0.0, 0.0, dt)

        pushed = math.hypot(pose[0] - self._push["start"][0],
                            pose[1] - self._push["start"][1])
        if pushed >= p["second_push_dist"]:
            self._front_obj = self._target
            self._front_obj["status"] = "captured"
            self._target = None
            self._events.append("SECOND_CAPTURED")
            self._plan_to_deposit(pose)
            self._set_state(TRANSPORT, t)
            return self.ctrl._limit(0.0, 0.0, dt)
        if t - self._push["t0"] > p["capture_push_limit_s"]:
            self._abandon_second(t, pose, "push_timeout")
            return self.ctrl._limit(0.0, 0.0, dt)
        herr = wrap_angle(self._push["yaw_lock"] - pose[2])
        return self.ctrl.straight(p["push_v"], dt, hold_yaw_err=herr)

    def _select_second(self, t, pose):
        """A 적재 직후, 입구에 물고 갈 B 선택: 가깝고, 보관함으로 가는 길이
        완만하게 꺾이고(입구 물체는 급회전에 빠짐), 시간이 남는 확정 타깃."""
        p = self.p
        if not p["double_carry"]:
            return None
        dep = (p["deposit_approach_x"], p["deposit_lane_y"])
        best, best_d = None, p["pair_max_dist"]
        for o in self.memory.objects:
            if o["status"] != "open" or not self._is_definite_target(o):
                continue
            d1 = math.hypot(o["x"] - pose[0], o["y"] - pose[1])
            if d1 >= best_d:
                continue
            a1 = math.atan2(o["y"] - pose[1], o["x"] - pose[0])
            a2 = math.atan2(dep[1] - o["y"], dep[0] - o["x"])
            if abs(wrap_angle(a2 - a1)) > math.radians(p["pair_max_turn_deg"]):
                continue
            d2 = math.hypot(dep[0] - o["x"], dep[1] - o["y"])
            need = (d1 + d2) / p["eff_speed"] + p["t_approach_est"] + \
                p["t_capture2_est"] + p["t_deposit_est"]
            if need > self.remaining(t) - 2.0:
                continue
            best, best_d = o, d1
        return best

    def _st_retreat(self, t, dt, pose, percep, ir):
        r = self._retreat
        if r.get("align") is not None:
            v, w, done = self.ctrl.rotate_to(pose, r["align"], dt)
            if done:
                r["align"] = None
                r["start"] = (pose[0], pose[1])
            return v, w
        moved = math.hypot(pose[0] - r["start"][0], pose[1] - r["start"][1])
        # 후진 방향이 벽/보관함이면 갈 수 있는 데까지만 물러난다
        bx = pose[0] - 0.18 * math.cos(pose[2])
        by = pose[1] - 0.18 * math.sin(pose[2])
        blocked = not self.geom.in_arena(bx, by, self.p["robot_radius"] * 0.8)
        if moved >= r["dist"] or blocked:
            then = r["then"]
            if then == GOTO and self._target is not None:
                if r.get("yaw_jitter"):
                    # 미세 시점 변경: 물체를 살짝 다른 각도에서 다시 본다
                    tgt = self._target
                    ang = math.atan2(pose[1] - tgt["y"], pose[0] - tgt["x"])
                    ang += r["yaw_jitter"]
                    d = self.p["standoff_dist"]
                    sx = tgt["x"] + d * math.cos(ang)
                    sy = tgt["y"] + d * math.sin(ang)
                    sx, sy = self._clamp_into_arena(sx, sy)
                    self._route = [(sx, sy)]
                else:
                    self._plan_to_standoff(pose, self._target)
                self._set_state(GOTO, t)
            elif then in (TOUR, GOTO, TRANSPORT):
                if then == TRANSPORT:
                    self._plan_to_deposit(pose)
                if then == TOUR:
                    self._target = None
                    self._cmds.append(dict(cmd="set_phase", phase="SEARCH"))
                self._set_state(then, t)
                if then == TOUR and r.get("escape_wp"):
                    self._tour_route = [r["escape_wp"]]   # 혼잡 구석 강제 이탈
            self._retreat = None
            return self.ctrl._limit(0.0, 0.0, dt)
        return self.ctrl.straight(-self.p["reverse_v"], dt, hold_yaw_err=0.0)

    def _st_transport(self, t, dt, pose, percep, ir):
        p = self.p
        # 운반 중 IR 유실 → 짧게 참았다가 회수 시도
        if not ir:
            if self._ir_lost_since is None:
                self._ir_lost_since = t
            elif t - self._ir_lost_since > p["ir_lost_patience_s"]:
                self._events.append("PAYLOAD_LOST")
                self._cmds.append(dict(cmd="note_payload_lost"))
                a = self._payload_obj
                # 물체는 대략 지금 로봇 앞에 남아 있다 — 위치 갱신 후 재접근.
                # 이미 verify 를 통과한 물체이므로 회수는 재검증 없이 진행한다.
                a["x"] = pose[0] + p["payload_offset"] * math.cos(pose[2])
                a["y"] = pose[1] + p["payload_offset"] * math.sin(pose[2])
                a["status"] = "open"
                if self._front_obj is not None:   # 입구 물체도 같이 흘렸다고 가정
                    f = self._front_obj
                    f["x"] = a["x"] + 0.09 * math.cos(pose[2])
                    f["y"] = a["y"] + 0.09 * math.sin(pose[2])
                    f["status"] = "open"
                    self._front_obj = None
                self._payload_obj = None
                self._target = a
                self._recovering = True
                self._begin_retreat(pose, p["retreat_dist"], then=GOTO)
                return self.ctrl._limit(0.0, 0.0, dt)
        else:
            self._ir_lost_since = None

        if not self._route:
            # 하역 정렬 지점 도착. 의도한 B 가 없으면 먼저 밀항 물체 털기.
            if self.p["shed_spin"] and self._front_obj is None:
                self._shed = dict(accum=0.0,
                                  goal=math.radians(self.p["shed_spin_angle_deg"]))
                self._events.append("SHED_SPIN")
                self._set_state(DEPOSIT_SHED, t)
            else:
                self._set_state(DEPOSIT_PUSH, t)
            return self.ctrl._limit(0.0, 0.0, dt)
        wp = self._route[0]
        last = len(self._route) == 1
        # 마지막(정렬) 지점에서는 -x 방향(보관함 쪽)을 보게 정렬
        v, w, done = self.ctrl.go_to(pose, wp, dt,
                                     final_yaw=math.pi if last else None)
        if done:
            self._route.pop(0)
        return v, w

    def _st_deposit_shed(self, t, dt, pose, percep, ir):
        """제자리 고속 회전으로 입구의 밀항 물체를 털어낸 뒤 서쪽(-x) 재정렬.

        A(빈 안쪽)는 옆벽·뒷벽에 갇혀 있어 제자리 회전에 안전하다. IR 이
        꺼지면 A 까지 빠진 것 — 운반 유실 플로우가 다음 틱에 처리한다.
        """
        p = self.p
        if not ir:   # 회전 중 A 까지 빠졌으면 운반 유실 플로우로
            self._set_state(TRANSPORT, t)
            return self.ctrl._limit(0.0, 0.0, dt)
        if self._shed["accum"] < self._shed["goal"]:
            v, w = self.ctrl._limit(0.0, p["shed_spin_w"], dt)
            self._shed["accum"] += abs(w) * dt
            return v, w
        v, w, done = self.ctrl.rotate_to(pose, math.pi, dt)
        if done:
            self._set_state(DEPOSIT_PUSH, t)
        return v, w

    def _st_deposit_push(self, t, dt, pose, percep, ir):
        p = self.p
        if self._front_obj is not None:
            depth = p["deposit_depth_double"]   # B 가 A 앞 ~0.09 에 놓인다
        else:
            depth = p["deposit_depths"][len(self.deposited) %
                                        len(p["deposit_depths"])]
        stop_x = depth + p["payload_offset"]
        if pose[0] <= stop_x or self.stall.update(dt, self._prev_v, pose):
            self._release = dict(start=(pose[0], pose[1]), t0=t)
            self._set_state(DEPOSIT_RELEASE, t)
            return self.ctrl._limit(0.0, 0.0, dt)
        herr = wrap_angle(math.pi - pose[2])
        # 보관함 근처(잔여 15cm)에서는 감속 — 정지 오버런으로 물체가 벽을 타는 것 방지
        v_dep = p["deposit_push_v"] if pose[0] > stop_x + 0.15 else p["push_v"]
        return self.ctrl.straight(v_dep, dt, hold_yaw_err=herr)

    def _st_deposit_release(self, t, dt, pose, percep, ir):
        p = self.p
        moved = math.hypot(pose[0] - self._release["start"][0],
                           pose[1] - self._release["start"][1])
        if moved >= p["release_reverse_dist"]:
            for obj in (self._payload_obj, self._front_obj):
                if obj is None:
                    continue
                obj["status"] = "deposited"
                self.deposited.append((obj.get("set"), obj.get("cls")))
                self.score += self._value(obj)
                self._events.append(
                    f"DEPOSITED({obj.get('set')}:{obj.get('cls')})")
            self._payload_obj = self._front_obj = None
            self._target = None
            self._cmds.append(dict(cmd="note_loaded", loaded=False))
            self._cmds.append(dict(cmd="reset_tracking"))
            self._cmds.append(dict(cmd="set_phase", phase="SEARCH"))
            self._set_state(TOUR, t)
            return self.ctrl._limit(0.0, 0.0, dt)
        # 후진 + 위글 (빈 벽 마찰로 물체가 딸려나오는 것 방지)
        wig = p["release_wiggle_w"] * math.sin(
            2 * math.pi * (t - self._release["t0"]) / p["release_wiggle_period_s"])
        v_rev = p["reverse_v"]
        if moved > 0.15:
            wig = 0.0                      # 물체와 떨어진 뒤에는 직선 후진
            v_rev = p["release_reverse_v"]  # 그리고 빠르게 이탈
        return self.ctrl._limit(-v_rev, wig, dt)

    def _st_park(self, t, dt, pose, percep, ir):
        # 보관함/스티커존 밖이면 그 자리에서 정지 대기
        if self.geom.sticker_zone.contains(pose[0], pose[1], margin=0.1):
            v, w, _ = self.ctrl.go_to(pose, (1.6, 0.8), dt)
            return v, w
        return self.ctrl._limit(0.0, 0.0, dt)

    # ---------------- 내부 유틸 ----------------

    def _set_state(self, s, t):
        if s != self.state:
            self._events.append(f"{self.state}->{s}")
            self.state = s
            self.stall.reset()
            if s == APPROACH:
                self._cmds.append(dict(cmd="set_phase", phase="VERIFY"))
            if s == TOUR:
                self._tour_route = []   # 위치가 바뀌었으니 레인 경로 재계획

    def _enter_approach(self, t, pose):
        # 유실물 회수도 APPROACH 를 거친다: 전방캠 조향 서보로 실물에 재정렬해야
        # 한다 (기억 속 위치로만 blind push 하면 옆으로 빠진 물체를 영영 놓침).
        # 분류 게이트는 _st_approach 에서 _recovering 이면 우회.
        self._approach_t0 = t
        self._approach_start = (pose[0], pose[1])
        self._vr_hist = []
        self._set_state(APPROACH, t)

    def _begin_retreat(self, pose, dist, then, yaw_jitter=0.0):
        self._retreat = dict(start=(pose[0], pose[1]), dist=dist, then=then,
                             yaw_jitter=yaw_jitter, align=None)
        # 후진 방향이 벽이면 그대로 후진 불가 (즉시 종료 → 전진 → 재스쿱
        # 무한루프). 먼저 기수를 경기장 중심 반대로 돌려 후진로를 연다.
        bx = pose[0] - 0.25 * math.cos(pose[2])
        by = pose[1] - 0.25 * math.sin(pose[2])
        if not self.geom.in_arena(bx, by, self.p["robot_radius"] * 0.8):
            cx = self.geom.arena_w / 2.0
            cy = self.geom.arena_h / 2.0
            self._retreat["align"] = wrap_angle(
                math.atan2(cy - pose[1], cx - pose[0]) + math.pi)
        self.ctrl.reset()
        self._set_state(RETREAT, self._last_t)

    def _face_target(self, pose):
        tgt = self._target
        return math.atan2(tgt["y"] - pose[1], tgt["x"] - pose[0])

    def _clamp_into_arena(self, x, y):
        m = self.p["robot_radius"] + 0.05
        return (min(self.geom.arena_w - m, max(m, x)),
                min(self.geom.arena_h - m, max(m, y)))

    def _build_tour(self):
        p = DEFAULT_PARAMS if not hasattr(self, "p") else self.p
        if getattr(self, "cube_hunt", False):
            # cube 공지 경기: 외곽 일주 — 모든 물체를 동서남북 4방향에서 관측해
            # 무지면 섹터 증명을 완성한다. 레인이 최외곽 배치열에 가깝지만
            # 경로계획이 알려진 물체를 국소 우회한다 (일부 물체는 벽쪽 면
            # 관측이 안 되면 확정 불가 — 그 큐브는 포기가 맞다).
            w, h = self.geom.arena_w, self.geom.arena_h
            return [(w - 0.35, 0.45), (w - 0.35, h - 0.4), (0.4, h - 0.4),
                    (0.4, 0.75)]
        x0, x1 = p["tour_margin_x"], self.geom.arena_w - p["tour_margin_x"]
        wps = []
        for i, y in enumerate(p["tour_lanes_y"]):
            if i % 2 == 0:
                wps += [(x1, y), (x0, y)]
            else:
                wps += [(x0, y), (x1, y)]
        return wps

    def _plan_to_standoff(self, pose, target):
        """접근 각도 선택: verify 캠 시야에 다른 물체가 겹치지 않는 방향을 고른다.

        (전방캠은 시야각 안 '가장 가까운' 물체를 검증하므로, 접근 축 근처에 남이
        있으면 엉뚱한 물체를 검증하거나 승인 없이 스쳐 담을 위험이 있다.)
        """
        d = self.p["standoff_dist"]
        base = math.atan2(pose[1] - target["y"], pose[0] - target["x"])
        obstacles = self.memory.obstacles(exclude_id=target["id"])
        keep = [self.geom.storage]
        # cube 인증 보완 방문: 아직 무지 확인이 안 된 각도(최대 갭의 중앙)에서
        # '보기만' 하면 된다 — 각도를 정확히 지키고(회랑 회피 불필요, 포획 안
        # 함) verify 가시거리 안(1.0m)까지만 간다.
        if self.cube_hunt and target.get("blank_sectors") and \
                not self._is_definite_target(target):
            base = self._gap_mid_angle(target)
            target["visit_sec"] = int((base + math.pi) /
                                      (2 * math.pi) * 16) % 16
            sx, sy = self._clamp_into_arena(target["x"] + 1.0 * math.cos(base),
                                            target["y"] + 1.0 * math.sin(base))
            route = self.planner.plan((pose[0], pose[1]), (sx, sy),
                                      obstacles, keep)
            self._route = route if route else [(sx, sy)]
            return

        def corridor_block(ang):
            sx = target["x"] + d * math.cos(ang)
            sy = target["y"] + d * math.sin(ang)
            ux, uy = target["x"] - sx, target["y"] - sy
            un = math.hypot(ux, uy)
            ux, uy = ux / un, uy / un
            n = 0
            for ox, oy in obstacles:
                s = (ox - sx) * ux + (oy - sy) * uy
                if -0.1 < s < d + 0.35:
                    perp = abs(-(oy - sy) * ux + (ox - sx) * uy)
                    # verify 캠은 '원뿔 안 최근접'을 판정한다: 접근 축에서
                    # 0.28m 안쪽 물체는 목표보다 가까워지는 순간 잠금을 가로챈다
                    if perp < 0.28:
                        n += 1
            return n

        best, best_cost = None, None
        for i, off in enumerate((0.0, 0.6, -0.6, 1.2, -1.2, 1.9, -1.9, math.pi)):
            ang = base + off
            sx = target["x"] + d * math.cos(ang)
            sy = target["y"] + d * math.sin(ang)
            sx, sy = self._clamp_into_arena(sx, sy)
            if self.geom.sticker_zone.contains(sx, sy):
                continue
            cost = corridor_block(ang) * 10 + i
            if best_cost is None or cost < best_cost:
                best, best_cost = (sx, sy), cost
            if cost == 0:
                break
        sx, sy = best if best else self._clamp_into_arena(
            target["x"] + d * math.cos(base), target["y"] + d * math.sin(base))
        route = self.planner.plan((pose[0], pose[1]), (sx, sy), obstacles, keep)
        self._route = route if route else [(sx, sy)]

    def _plan_to_deposit(self, pose):
        p = self.p
        goal = (p["deposit_approach_x"], p["deposit_lane_y"])
        obstacles = self.memory.obstacles(
            exclude_id=self._target["id"] if self._target else None)
        route = self.planner.plan((pose[0], pose[1]), goal, obstacles, ())
        self._route = route if route else [goal]

    def _matches_target(self, obj, set_name):
        """cls/target 둘 다 실제 값일 때만 매칭 (None==None 오인 방지 —
        세트별 단독 경기에서는 targets 에 한 세트만 들어온다)."""
        cls, tgt = obj.get("cls"), self.targets.get(set_name)
        return (obj.get("set") == set_name and cls is not None
                and tgt is not None and cls == tgt)

    def _gap_mid_angle(self, ent):
        """무지면 섹터의 최대 빈 구간 중앙 방향 (물체→관측점 각)."""
        occ = sorted(ent["blank_sectors"])
        if not occ:
            return 0.0
        best_gap, best_mid = -1, 0.0
        for i, s in enumerate(occ):
            nxt = occ[(i + 1) % len(occ)]
            gap = (nxt - s) % 16 or 16
            if gap > best_gap:
                best_gap = gap
                mid = (s + gap / 2.0) % 16
                best_mid = -math.pi + (mid + 0.5) * (2 * math.pi / 16)
        return best_mid

    def _value(self, obj):
        if self._matches_target(obj, "set2"):
            return self.p["value_set2"]
        if self._matches_target(obj, "set1"):
            return self.p["value_set1"]
        return self.p["value_unknown"]

    def _is_definite_target(self, obj):
        return obj["rank"] >= 3 and (self._matches_target(obj, "set1") or
                                     self._matches_target(obj, "set2"))

    def _est_trip_time(self, pose, obj):
        p = self.p
        d1 = math.hypot(obj["x"] - pose[0], obj["y"] - pose[1])
        dep = (p["deposit_approach_x"], p["deposit_lane_y"])
        d2 = math.hypot(obj["x"] - dep[0], obj["y"] - dep[1])
        return (d1 + d2) / p["eff_speed"] + p["t_approach_est"] + \
            p["t_capture_est"] + p["t_deposit_est"]

    def _select_target(self, t, pose):
        """가치/시간 탐욕 + 시간 컷오프. 확정 타깃 > 미확인 후보."""
        best, best_score = None, 0.0
        rem = self.remaining(t)
        for o in self.memory.objects:
            if o["status"] not in ("open",):
                # cube 공지 경기: 보류(defer)된 큐브는 부족한 섹터 방향에서
                # 다시 봐야 인증이 진행된다 — 제한 횟수 내 보완 방문 허용
                if not (self.cube_hunt and o["status"] == "defer"
                        and o.get("visits", 0) <= 6):
                    continue
            if o["rank"] < 1:      # 한 번 스친 관측만으로는 안 움직임
                continue
            # 확정 비타깃(목표 아닌 형상/과일로 분류 완료)은 제외
            if o["rank"] >= 3 and not self._is_definite_target(o):
                continue
            definite = self._is_definite_target(o)
            # 미확인 조사 허용 시점 (unknown_gate): 확정 타깃 우선이 원칙이지만,
            # 목표가 미확인 속에 숨는 경기(세트2: 과일면이 레인에서 안 보임)에서는
            # 조사를 앞당기는 것이 유리할 수 있다 — sim_speed_sweep.py 로 비교.
            gate = self.p["unknown_gate"]
            unknown_ok = (self._tour_pass_done or gate == "always" or
                          (gate == "first_lane" and self._tour_idx >= 2))
            if not definite and not unknown_ok:
                continue
            trip = self._est_trip_time(pose, o)
            if trip > rem - self.p["endgame_margin_s"]:
                continue
            val = self._value(o) if definite else self.p["value_unknown"]
            score = val / trip
            if score > best_score:
                best, best_score = o, score
        return best

    def _blacklist(self, obj, reason):
        if obj is not None:
            obj["status"] = "blacklist"
            self._events.append(f"BLACKLIST({reason})")
        self._recovering = False

    def _dbg(self):
        return dict(state=self.state, percep_cmds=self._cmds,
                    events=self._events, score=self.score,
                    deposited=list(self.deposited),
                    target=self._target["id"] if self._target else None,
                    payload=self._payload_obj["id"] if self._payload_obj else None,
                    front=self._front_obj["id"] if self._front_obj else None,
                    route=list(self._route))
