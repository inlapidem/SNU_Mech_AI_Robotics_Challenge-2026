"""주행 코어 (ROS 의존성 없음 — WSL 시뮬레이션 검증용).

wall_localizer_core 와 같은 원칙: 알고리즘은 전부 여기에 두고, ROS 노드는 입출력만
담당한다. 좌표계는 localization/README.md 와 동일 (map 원점 = 벽 안쪽 왼쪽-아래,
x→오른쪽, y→위, yaw는 +x 기준 반시계).

구성:
  Rect / ArenaGeometry   경기장·스타트·보관함·스티커존 사각형
  CamSpec / bearing_range_from_bbox   단안 거리(물체 높이 8cm 고정) + 방위각
  DiffDriveController    회전-후-직진 P 제어기 + 가감속 제한 (go_to / rotate_to)
  GridPlanner            10cm 격자 A* + 시선 단축(shortcut) — 관측된 물체 회피
  StallDetector          명령 대비 실제 이동 감시 (부딪힘/걸림 감지)
"""

import heapq
import math
from dataclasses import dataclass, field


def wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# ---------------------------------------------------------------- 구역(사각형)

@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x, y, margin=0.0):
        return (self.x0 - margin <= x <= self.x1 + margin and
                self.y0 - margin <= y <= self.y1 + margin)

    def center(self):
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def dist(self, x, y):
        dx = max(self.x0 - x, 0.0, x - self.x1)
        dy = max(self.y0 - y, 0.0, y - self.y1)
        return math.hypot(dx, dy)


@dataclass
class ArenaGeometry:
    """룰 기반 고정 기하 (2026-07 룰: 4x4m, 스타트 우하단 40cm, 보관함 좌하단 40cm)."""
    arena_w: float = 4.0
    arena_h: float = 4.0
    start_zone: Rect = field(default_factory=lambda: Rect(3.6, 0.0, 4.0, 0.4))
    storage: Rect = field(default_factory=lambda: Rect(0.0, 0.0, 0.4, 0.4))
    # 태극기 스티커 오탐 지오펜스: 스티커는 보관함 몸체에 붙으므로 이 안으로
    # 투영되는 관측은 무시한다. (배치 격자는 벽에서 0.5m 이상이라 실물과 안 겹침)
    sticker_zone: Rect = field(default_factory=lambda: Rect(0.0, 0.0, 0.55, 0.55))

    def in_arena(self, x, y, margin):
        return (margin <= x <= self.arena_w - margin and
                margin <= y <= self.arena_h - margin)


# ------------------------------------------------- 단안 거리/방위 (물체 높이 8cm 고정)

@dataclass(frozen=True)
class CamSpec:
    """카메라 1대의 장착/광학 파라미터 (실측 후 params.yaml 로 조정)."""
    name: str
    yaw_deg: float          # base_link 전방 기준 광축 방향 (좌측캠 +90 등)
    hfov_deg: float         # 수평 화각
    img_w: int
    img_h: int

    @property
    def f_px(self):
        return (self.img_w / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)


OBJ_HEIGHT_M = 0.08   # 룰: 모든 물체는 눕혔을 때 높이 8cm


def bearing_range_from_bbox(cam: CamSpec, bbox):
    """bbox=(x0,y0,x1,y1) 픽셀 → (로봇 기준 방위각 rad, 거리 m).

    거리는 '모든 물체 높이 8cm' 룰을 이용한 단안 추정: range = f * H / h_px.
    원거리(작은 bbox)에서 ±20% 수준 — 경로계획/우선순위용이지 정밀 접근용이 아니다.
    정밀 접근은 verify 캠 조향(steering)이 담당한다.
    """
    x0, y0, x1, y1 = bbox
    h_px = max(1.0, y1 - y0)
    cx = (x0 + x1) / 2.0
    rng = cam.f_px * OBJ_HEIGHT_M / h_px
    bearing_cam = -math.atan2(cx - cam.img_w / 2.0, cam.f_px)  # 이미지 오른쪽 = 음(-) 방위
    return wrap_angle(math.radians(cam.yaw_deg) + bearing_cam), rng


def project_to_map(pose, bearing, rng):
    """로봇 자세 (x,y,yaw) + 로봇 기준 방위/거리 → map 좌표."""
    x, y, yaw = pose
    a = yaw + bearing
    return x + rng * math.cos(a), y + rng * math.sin(a)


# ------------------------------------------------------------- 차동구동 제어기

@dataclass
class ControllerConfig:
    max_v: float = 0.15          # 순항 속도 [m/s] (max_wheel_speed 실측 전 보수값)
    max_w: float = 1.0           # 최대 각속도 [rad/s]
    accel: float = 0.4           # 선속도 가감속 [m/s^2]
    ang_accel: float = 3.0       # 각가속도 [rad/s^2]
    kp_lin: float = 1.2          # 거리 P 이득
    kp_ang: float = 2.5          # 방향 P 이득
    turn_in_place_deg: float = 50.0   # 방향 오차가 이보다 크면 제자리 회전
    decel_dist: float = 0.35     # 목표 전 감속 시작 거리
    pos_tol: float = 0.06        # 도달 판정 [m]
    yaw_tol_deg: float = 5.0     # 방향 도달 판정
    min_v: float = 0.04          # 정지마찰 대비 최저 유효 선속도
    min_w: float = 0.25          # 최저 유효 각속도


class DiffDriveController:
    """(v, w) 를 내는 자세 제어기. step(dt) 마다 가감속 제한을 적용한다."""

    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.v = 0.0
        self.w = 0.0

    def reset(self):
        self.v = self.w = 0.0

    def _limit(self, v_des, w_des, dt):
        c = self.cfg
        dv = max(-c.accel * dt, min(c.accel * dt, v_des - self.v))
        dw = max(-c.ang_accel * dt, min(c.ang_accel * dt, w_des - self.w))
        self.v += dv
        self.w += dw
        return self.v, self.w

    def rotate_to(self, pose, target_yaw, dt):
        """제자리 회전. returns (v, w, done)"""
        c = self.cfg
        err = wrap_angle(target_yaw - pose[2])
        if abs(err) < math.radians(c.yaw_tol_deg):
            v, w = self._limit(0.0, 0.0, dt)
            return v, w, abs(self.w) < 0.05
        w_des = max(-c.max_w, min(c.max_w, c.kp_ang * err))
        if abs(w_des) < c.min_w:
            w_des = math.copysign(c.min_w, w_des)
        v, w = self._limit(0.0, w_des, dt)
        return v, w, False

    def go_to(self, pose, goal_xy, dt, final_yaw=None, speed_cap=None):
        """목표점 주행 (회전-후-직진 + 근접 감속). returns (v, w, done)"""
        c = self.cfg
        x, y, yaw = pose
        dx, dy = goal_xy[0] - x, goal_xy[1] - y
        dist = math.hypot(dx, dy)

        if dist < c.pos_tol:
            if final_yaw is not None:
                return self.rotate_to(pose, final_yaw, dt)
            v, w = self._limit(0.0, 0.0, dt)
            return v, w, abs(self.v) < 0.02 and abs(self.w) < 0.05

        heading = math.atan2(dy, dx)
        herr = wrap_angle(heading - yaw)
        if abs(herr) > math.radians(c.turn_in_place_deg):
            w_des = max(-c.max_w, min(c.max_w, c.kp_ang * herr))
            if abs(w_des) < c.min_w:
                w_des = math.copysign(c.min_w, w_des)
            v, w = self._limit(0.0, w_des, dt)
            return v, w, False

        cap = c.max_v if speed_cap is None else min(c.max_v, speed_cap)
        v_des = min(cap, c.kp_lin * dist,
                    cap * max(0.25, min(1.0, dist / c.decel_dist)))
        v_des = max(c.min_v, v_des)
        # 방향 오차가 클수록 감속 (곡률 완화)
        v_des *= max(0.2, math.cos(herr))
        w_des = max(-c.max_w, min(c.max_w, c.kp_ang * herr))
        v, w = self._limit(v_des, w_des, dt)
        return v, w, False

    def straight(self, v_des, dt, hold_yaw_err=None, kp_hold=2.0):
        """직진/후진 (BLIND 푸시·하역 이탈용). hold_yaw_err 지정 시 방향 유지."""
        w_des = 0.0
        if hold_yaw_err is not None:
            w_des = max(-0.6, min(0.6, kp_hold * hold_yaw_err))
        return self._limit(v_des, w_des, dt)


# ------------------------------------------------------------------ A* 경로계획

class GridPlanner:
    """관측된 물체(8cm, 라이다에 안 보임)를 피해 가는 10cm 격자 A*.

    라이다 스캔 평면이 물체 위를 지나므로 장애물 지도는 오직 카메라 관측 누적으로
    만든다. 못 본 물체와의 충돌은 StallDetector 가 잡아서 임시 장애물로 등록한다.
    """

    def __init__(self, geom: ArenaGeometry, cell=0.10, robot_radius=0.22,
                 obj_radius=0.06, extra_margin=0.04):
        self.geom = geom
        self.cell = cell
        self.inflate = robot_radius + obj_radius + extra_margin
        self.wall_margin = robot_radius + 0.02
        self.nx = int(round(geom.arena_w / cell))
        self.ny = int(round(geom.arena_h / cell))

    # --- 격자 변환 ---
    def _to_cell(self, x, y):
        return (min(self.nx - 1, max(0, int(x / self.cell))),
                min(self.ny - 1, max(0, int(y / self.cell))))

    def _to_xy(self, c):
        return ((c[0] + 0.5) * self.cell, (c[1] + 0.5) * self.cell)

    def _blocked(self, cx, cy, obstacles, keepouts):
        x, y = (cx + 0.5) * self.cell, (cy + 0.5) * self.cell
        if not self.geom.in_arena(x, y, self.wall_margin):
            return True
        for r in keepouts:
            if r.contains(x, y, margin=self.wall_margin):
                return True
        for ox, oy in obstacles:
            if (x - ox) ** 2 + (y - oy) ** 2 < self.inflate ** 2:
                return True
        return False

    def _los_free(self, p0, p1, obstacles, keepouts):
        d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        n = max(2, int(d / (self.cell * 0.5)))
        for i in range(n + 1):
            t = i / n
            x = p0[0] + t * (p1[0] - p0[0])
            y = p0[1] + t * (p1[1] - p0[1])
            c = self._to_cell(x, y)
            if self._blocked(c[0], c[1], obstacles, keepouts):
                return False
        return True

    def plan(self, start_xy, goal_xy, obstacles, keepouts=()):
        """A* → 웨이포인트 목록 (goal 포함, start 제외). 실패 시 None.

        obstacles: [(x,y), ...] — 회피할 물체 중심들 (목표 물체는 빼고 넣을 것)
        keepouts:  [Rect, ...]  — 진입 금지 구역 (예: 하역 전 보관함)
        """
        start = self._to_cell(*start_xy)
        goal = self._to_cell(*goal_xy)

        # 목표/출발 셀이 팽창 반경에 걸려 막혀 있으면 그 근처 물체는 무시하고 연다
        # (접근 목표는 물체 바로 앞이므로 당연히 걸린다).
        obs_goal = [o for o in obstacles
                    if (o[0] - goal_xy[0]) ** 2 + (o[1] - goal_xy[1]) ** 2
                    > self.inflate ** 2 * 0.9]
        obs = [o for o in obs_goal
               if (o[0] - start_xy[0]) ** 2 + (o[1] - start_xy[1]) ** 2
               > self.inflate ** 2 * 0.9]

        if self._blocked(goal[0], goal[1], obs, keepouts):
            # 목표 셀이 막혔으면(장애물 밀집·가상 장애물 축적) 근처의 열린
            # 셀로 완화 — None 을 돌려주면 호출부가 직선 폴백으로 장애물을
            # 관통하므로(재스쿱 루프) 완화가 훨씬 안전하다.
            found = None
            for ring in range(1, 6):
                for dx in range(-ring, ring + 1):
                    for dy in range(-ring, ring + 1):
                        if max(abs(dx), abs(dy)) != ring:
                            continue
                        c = (goal[0] + dx, goal[1] + dy)
                        if not (0 <= c[0] < self.nx and 0 <= c[1] < self.ny):
                            continue
                        if not self._blocked(c[0], c[1], obs, keepouts):
                            found = c
                            break
                    if found:
                        break
                if found:
                    break
            if found is None:
                return None
            goal = found
            goal_xy = self._to_xy(goal)

        openq = [(0.0, start)]
        g = {start: 0.0}
        came = {}
        seen = set()
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == goal:
                break
            if cur in seen:
                continue
            seen.add(cur)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (cur[0] + dx, cur[1] + dy)
                    if not (0 <= nb[0] < self.nx and 0 <= nb[1] < self.ny):
                        continue
                    if self._blocked(nb[0], nb[1], obs, keepouts):
                        continue
                    step = math.hypot(dx, dy) * self.cell
                    ng = g[cur] + step
                    if ng < g.get(nb, 1e18):
                        g[nb] = ng
                        came[nb] = cur
                        h = math.hypot(goal[0] - nb[0], goal[1] - nb[1]) * self.cell
                        heapq.heappush(openq, (ng + h, nb))
        if goal not in came and goal != start:
            return None

        cells = [goal]
        while cells[-1] != start:
            cells.append(came[cells[-1]])
        cells.reverse()
        pts = [self._to_xy(c) for c in cells[1:]]
        if not pts:
            return [goal_xy]
        pts[-1] = goal_xy

        # 시선 단축: 볼 수 있는 가장 먼 웨이포인트로 건너뛰기
        out = []
        cur = start_xy
        i = 0
        while i < len(pts):
            j = len(pts) - 1
            while j > i and not self._los_free(cur, pts[j], obs, keepouts):
                j -= 1
            out.append(pts[j])
            cur = pts[j]
            i = j + 1
        return out


# ------------------------------------------------------------------ 스탈 감지

class StallDetector:
    """명령 속도 대비 오도메트리 이동이 계속 부족하면 '걸림'으로 판정.

    라이다에 안 보이는 물체/벽 접촉의 최후 감지선. 감지되면 미션이 후진+임시
    장애물 등록으로 대응한다.
    """

    def __init__(self, window_s=0.8, ratio=0.25, min_cmd_v=0.03):
        self.window_s = window_s
        self.ratio = ratio
        self.min_cmd_v = min_cmd_v
        self.reset()

    def reset(self):
        self._acc_t = 0.0
        self._acc_cmd = 0.0
        self._acc_move = 0.0
        self._last_xy = None

    def update(self, dt, cmd_v, pose):
        xy = (pose[0], pose[1])
        moved = 0.0
        if self._last_xy is not None:
            moved = math.hypot(xy[0] - self._last_xy[0], xy[1] - self._last_xy[1])
        self._last_xy = xy

        if abs(cmd_v) < self.min_cmd_v:
            self._acc_t = self._acc_cmd = self._acc_move = 0.0
            return False

        self._acc_t += dt
        self._acc_cmd += abs(cmd_v) * dt
        self._acc_move += moved
        if self._acc_t < self.window_s:
            return False
        stalled = self._acc_move < self.ratio * self._acc_cmd
        self._acc_t = self._acc_cmd = self._acc_move = 0.0
        return stalled
