#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""explore_demo 폐루프 시뮬레이터 — 하드웨어 없이 탐사 로직 전체를 검증한다.

왜 필요한가: explore_demo 의 실패는 대부분 '한 프레임의 계산'이 아니라 루프가
서로를 먹여 살리는 방식에서 나온다 (목표가 없어 정지 → 정지해서 맵이 안 바뀜 →
목표가 영영 안 생김). 저장된 map.npy 를 리플레이하는 것만으로는 이런 자기고착을
못 잡는다. 여기서는 진짜 세계(4x4 아레나 + 벽 틈 + 바깥 실험실)를 두고 라이다
레이캐스팅 → 점유격자 누적 → explore_demo 주행 → 로봇 이동을 닫아서 돌린다.
가상시계라 200초 주행이 실제 5~10초에 끝난다.

실제로 이 하네스가 잡은 것:
  * 아레나 박스 오추정 시 통과영역이 5047→72셀로 붕괴하고 로봇이 '목표없음'
    상태로 영구 정지 (→ ArenaBox.sanity + 목표 공백 워치독 추가)
  * ESCAPE 회복 우선순위(전진→회전→후진) 및 후진 사용률

사용:
  python3 sim/explore_sim.py            # 주행 시나리오 4종 + 회복 우선순위 단위검증
  python3 sim/explore_sim.py --drive    # 주행만
  python3 sim/explore_sim.py --unit     # 회복 우선순위만
"""
import contextlib
import io as _io
import math
import os
import sys
import time as real_time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import deployment.explore_demo as ex          # noqa: E402
from deployment.explore_demo import Roam, front_clearance   # noqa: E402

RES = 0.05
ORIGIN = (-10.0, -10.0)
N = 400                                        # 20m x 20m truth grid
ARENA_HALF = 2.0                               # 4x4 m 경기장

# ---- 구동계 모델 (전부 가정값 — 실측되면 갱신할 것) ----
K_PWM = 0.32 / 110.0     # PWM → 정상속도 [m/s]. pwm 110 ≈ 0.32 m/s 로 가정
TRACK_W = 0.30           # 트랙폭 [m]
TAU = 0.25               # 구동계 1차 지연 시정수 [s] → 제동거리 ≈ v·TAU


# ------------------------------------------------------------------ 진짜 세계
# 아레나 내부 장애물 (x, y, 반지름[m]).
# ★ 빈 아레나로는 고속 설정을 판별할 수 없다 — 실측: pwm 110/255, 조향 포화
# 유무, 램프 유무 6개 조합이 **전부 충돌 0** 이었다. 벽만 있고 안이 비면 회전반경이
# 0.85m 든 1.55m 든 부딪힐 것이 없기 때문이다. 실제 경기장에는 물체가 놓이므로
# 기둥을 넣어 '좁은 틈 통과 + 급선회'가 실제로 요구되게 만든다.
OBSTACLES = [
    (0.55, 0.60, 0.16),
    (-0.75, 0.35, 0.14),
    (0.10, -0.85, 0.15),
    (-0.60, -1.05, 0.12),
    (1.15, -0.35, 0.13),
]


def build_truth(obstacles=True):
    t = np.zeros((N, N), bool)

    def wall(x0, y0, x1, y1):
        n = int(max(abs(x1 - x0), abs(y1 - y0)) / (RES * 0.5)) + 1
        for i in range(n + 1):
            x = x0 + (x1 - x0) * i / n
            y = y0 + (y1 - y0) * i / n
            t[int((y - ORIGIN[1]) / RES), int((x - ORIGIN[0]) / RES)] = True

    a = ARENA_HALF
    # 경기장 외벽. 오른쪽 벽에 0.25m 틈 — 라이다는 새 나가지만(맵 오염) 로봇은
    # 못 나간다. 실측 로그(0719_194434)에서 맵의 76%가 이런 누출이었다.
    wall(-a, -a, a, -a); wall(-a, a, a, a); wall(-a, -a, -a, a)
    wall(a, -a, a, 0.15); wall(a, 0.40, a, a)
    # 바깥 실험실 — 누출된 레이가 보게 될 먼 벽
    wall(-9, -8, 9, -8); wall(9, -8, 9, 8); wall(-9, 8, 9, 8)
    if obstacles:
        ys, xs = np.mgrid[0:N, 0:N]
        wx = ORIGIN[0] + (xs + 0.5) * RES
        wy = ORIGIN[1] + (ys + 0.5) * RES
        for ox, oy, r in OBSTACLES:
            t |= (wx - ox) ** 2 + (wy - oy) ** 2 <= r * r
    return t


TRUTH = build_truth(obstacles=os.environ.get("SIM_EMPTY") != "1")


def raycast(x, y, yaw, n_ray=360, rmax=12.0):
    """sllidar 흉내 — 히트 없으면 inf (실장비의 무효 레이와 같은 표현)."""
    b = np.linspace(-math.pi, math.pi, n_ray, endpoint=False)
    steps = np.arange(0.05, rmax, 0.02)
    px = x + np.cos(b + yaw)[:, None] * steps[None, :]
    py = y + np.sin(b + yaw)[:, None] * steps[None, :]
    gx = ((px - ORIGIN[0]) / RES).astype(np.int32).clip(0, N - 1)
    gy = ((py - ORIGIN[1]) / RES).astype(np.int32).clip(0, N - 1)
    hit = TRUTH[gy, gx]
    idx = np.where(hit.any(1), hit.argmax(1), -1)
    return b.astype(np.float32), np.where(idx >= 0, steps[idx.clip(0)],
                                          np.inf).astype(np.float32)


class MapBuilder:
    """slam_toolbox 흉내 — 스캔을 누적해 int8 점유격자를 만든다 (자세는 참값).
    ★ integrate 에 넘기는 방위는 반드시 월드 방위(bearing + yaw)여야 한다.
    로봇 방위를 그대로 넘기면 맵이 로봇 yaw 만큼 회전해 쌓여, 축정렬 아레나가
    기울어진 사각형으로 나온다 (이 하네스를 처음 짤 때 실제로 낸 실수)."""

    def __init__(self):
        self.g = np.full((N, N), -1, np.int8)

    def integrate(self, x, y, b_world, r):
        rr = np.where(np.isfinite(r), r, 12.0)
        for i in range(len(b_world)):
            d = rr[i]
            s = np.arange(0.0, d, RES * 0.4)
            gx = ((x + np.cos(b_world[i]) * s - ORIGIN[0]) / RES).astype(np.int32)
            gy = ((y + np.sin(b_world[i]) * s - ORIGIN[1]) / RES).astype(np.int32)
            gx = gx.clip(0, N - 1); gy = gy.clip(0, N - 1)
            self.g[gy, gx] = np.where(self.g[gy, gx] < 0, 0, self.g[gy, gx])
            if np.isfinite(r[i]):
                hx = min(max(int((x + math.cos(b_world[i]) * d - ORIGIN[0]) / RES), 0), N - 1)
                hy = min(max(int((y + math.sin(b_world[i]) * d - ORIGIN[1]) / RES), 0), N - 1)
                self.g[hy, hx] = 100

    def crop(self):
        ys, xs = np.where(self.g >= 0)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        return (self.g[y0:y1, x0:x1].copy(), RES,
                (ORIGIN[0] + x0 * RES, ORIGIN[1] + y0 * RES))


class Clock:
    """가상시계 — explore_demo 의 time 모듈을 통째로 대체해 실시간 대기를 없앤다."""

    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def sleep(self, d):
        self.t += max(0.0, d)

    def strftime(self, f, *a):
        return real_time.strftime(f, *a)


class World:
    """RosIO 대역 + 차동구동 물리. explore_demo 는 이걸 ROS 스택으로 착각한다."""

    MIN_RANGE = 0.0           # RosIO 기본값과 동일 (근접 반사 차단 끔)

    def __init__(self, x, y, yaw, clk):
        self.x, self.y, self.yaw, self.clk = x, y, yaw, clk
        self.n_near = 0       # RosIO 대역이므로 같은 진단 속성을 갖춰야 한다
        self.mb = MapBuilder()
        self.t_map = -1e9
        self.map = self.pose = self.slam_pose = self.scan = None
        self.node = None
        self.path = []
        self.bumps = 0
        self.v = self.w = 0.0      # 현재 실제 속도 (명령이 아니라 관성 반영값)
        self.vmax = 0.0
        self.refresh()

    def refresh(self):
        b, r = raycast(self.x, self.y, self.yaw)
        self.n_near = int((np.isfinite(r) & (r > 0) & (r < 0.45)).sum())
        if self.MIN_RANGE > 0:             # RosIO._on_scan 과 같은 처리
            r = np.where(np.isfinite(r) & (r > 0) & (r < self.MIN_RANGE),
                         np.inf, r)
        self.scan = (self.clk.time(), b, r)
        self.pose = self.slam_pose = (self.clk.time(), (self.x, self.y, self.yaw))
        if self.clk.time() - self.t_map > 2.0:      # map_update_interval 2.0s
            self.t_map = self.clk.time()
            self.mb.integrate(self.x, self.y, b + self.yaw, r)
            self.map = (self.clk.time(), *self.mb.crop())

    def step(self, pl, pr, dt):
        """PWM → 속도는 1차 지연(시정수 TAU)으로 — 관성/제동거리를 만든다.

        ★ 순수 기구학(명령 PWM 이 곧 속도)으로 두면 '정지 명령 = 즉시 정지'가
        되어 아무리 빨라도 충돌이 안 난다. 그러면 최고속 튜닝에 이 시뮬을
        전혀 쓸 수 없다. 1차 지연을 넣어야 속도 v 에서 대략 v·TAU 의 제동거리가
        생겨 감속 램프의 효과를 실제로 측정할 수 있다.

        K_PWM 과 TAU 는 **실측이 아니라 가정**이다 (소형 기어드 DC + 부하 기준).
        따라서 절대 충돌 수보다 '설정 간 상대 추세'를 읽는 용도로 쓸 것."""
        vl, vr = pl * K_PWM, pr * K_PWM
        v_cmd, w_cmd = (vl + vr) / 2.0, (vr - vl) / TRACK_W
        a = 1.0 - math.exp(-max(dt, 1e-3) / TAU)
        self.v += (v_cmd - self.v) * a
        self.w += (w_cmd - self.w) * a
        nx = self.x + self.v * math.cos(self.yaw) * dt
        ny = self.y + self.v * math.sin(self.yaw) * dt
        gx = int((nx - ORIGIN[0]) / RES); gy = int((ny - ORIGIN[1]) / RES)
        k4 = int(0.20 / RES)                        # 몸 반폭 안에 벽 → 접촉
        if TRUTH[gy - k4:gy + k4 + 1, gx - k4:gx + k4 + 1].any():
            self.bumps += 1
            self.v *= 0.3                           # 접촉 = 급감속
        else:
            self.x, self.y = nx, ny
        self.yaw = ex.wrap_pi(self.yaw + self.w * dt)
        self.path.append((self.x, self.y))
        self.vmax = max(self.vmax, abs(self.v))
        self.refresh()

    def try_reset_origin(self, timeout=2.0):
        return False


def drive(x, y, yaw, out_dir=None, max_secs=200.0, extra=()):
    """한 시나리오를 끝까지 돌리고 (World, stdout) 반환."""
    clk = Clock()
    W = World(x, y, yaw, clk)
    saved = (ex.time, ex.rclpy, ex.RosIO, Roam.step)
    ex.time = clk
    ex.rclpy = type("R", (), {"init": staticmethod(lambda: None),
                              "shutdown": staticmethod(lambda: None)})()
    ex.RosIO = lambda *a, **k: W
    last = [clk.time()]
    orig = Roam.step

    def step(self, t, pose, b, r, goal):
        out = orig(self, t, pose, b, r, goal)
        dt = clk.time() - last[0]
        last[0] = clk.time()
        W.step(out[0], out[1], min(dt, 0.2))
        return out
    Roam.step = step
    sys.argv = ["explore_demo", "--dry-run", "--no-cam", "--no-reset",
                "--max-secs", str(max_secs)] + list(extra)
    if out_dir:
        sys.argv += ["--out", out_dir]
    buf = _io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ex.main()
    except SystemExit as e:
        buf.write(f"SystemExit: {e}\n")
    finally:
        ex.time, ex.rclpy, ex.RosIO, Roam.step = saved
    return W, buf.getvalue()


# ------------------------------------------------------------------ 시나리오
# 시작 지점은 아레나 오른쪽 아래 코너 (실경기 배치) — 시작 즉시 두 벽이 가깝다.
SCENARIOS = [
    ("코너/벽 정면(+x)", 1.55, -1.55, 0.0),
    ("코너/벽 정면(-y)", 1.55, -1.55, -math.pi / 2),
    ("코너/안쪽(135°)", 1.55, -1.55, 2.36),
    ("중앙/임의", 0.6, 0.7, 0.5),
]


def run_drive():
    ok = True
    for name, x, y, yaw in SCENARIOS:
        cnt = {"rev": 0, "fwd": 0, "turn": 0, "stop": 0, "n": 0}
        orig = Roam.step

        def step(self, t, pose, b, r, goal, _o=orig):
            out = _o(self, t, pose, b, r, goal)
            pl, pr = out[0], out[1]
            cnt["n"] += 1
            if pl < -5 and pr < -5:
                cnt["rev"] += 1
            elif pl > 5 and pr > 5:
                cnt["fwd"] += 1
            elif abs(pl) > 5 or abs(pr) > 5:
                cnt["turn"] += 1
            else:
                cnt["stop"] += 1
            return out
        Roam.step = step
        try:
            W, out = drive(x, y, yaw)
        finally:
            Roam.step = orig
        lock = [l for l in out.splitlines() if "아레나 인식" in l]
        end = [l for l in out.splitlines() if "종료 (" in l]
        dist = sum(math.hypot(W.path[i + 1][0] - W.path[i][0],
                              W.path[i + 1][1] - W.path[i][1])
                   for i in range(len(W.path) - 1))
        n = max(cnt["n"], 1)
        good = W.bumps == 0 and dist > 8.0 and bool(lock)
        ok &= good
        print(f"  [{'OK ' if good else 'FAIL'}] {name}")
        print(f"        {lock[0].strip() if lock else '아레나 인식 실패!'}")
        print(f"        {end[0].strip() if end else '종료 로그 없음'}")
        print(f"        주행 {dist:.1f}m · 벽 접촉 {W.bumps} · 최고 {W.vmax:.2f}m/s · "
              f"전진 {100 * cnt['fwd'] // n}% 회전 {100 * cnt['turn'] // n}% "
              f"후진 {100 * cnt['rev'] // n}%")
    return ok


def run_unit():
    """ESCAPE 회복 우선순위: 전진 가능? → 회전 가능? → 후진 가능? 순서 검증."""
    B = np.linspace(-math.pi, math.pi, 360, endpoint=False).astype(np.float32)

    def scan(front=3.0, side=3.0, rear=3.0, spike=None):
        r = np.full(360, side, np.float32)
        r[np.abs(B) <= math.radians(45)] = front
        r[np.abs(np.abs(B) - math.pi) <= math.radians(45)] = rear
        if spike is not None:
            r[200] = spike                 # 임의 방향 근접 장애물 → 회전 불가
        return r

    cases = [
        ("앞 열림", dict(front=3.0), "FWD"),
        ("앞 막힘 · 옆 열림", dict(front=0.35), "TURN"),
        ("앞 막힘 · 회전 불가 · 뒤 열림", dict(front=0.35, spike=0.25), "REV"),
        ("앞·뒤 막힘 · 회전 불가", dict(front=0.35, rear=0.4, spike=0.25), None),
        ("앞·뒤 막힘 · 회전 가능", dict(front=0.35, rear=0.4), "TURN"),
    ]
    ok = True
    for name, kw, want in cases:
        r = scan(**kw)
        got = Roam()._esc_pick(front_clearance(B, r), B, r)
        ok &= got == want
        print(f"  [{'OK ' if got == want else 'FAIL'}] {name} -> {got} (기대 {want})")

    r = scan(front=0.35, spike=0.25)
    got = Roam()._esc_pick(front_clearance(B, r), B, r, avoid_rev=True)
    ok &= got != "REV"
    print(f"  [{'OK ' if got != 'REV' else 'FAIL'}] 직전 후진이면 재후진 금지 -> {got}")

    roam = Roam(); roam.trigger_escape(0.0)
    seq, r = [], scan(front=0.35)
    for i in range(40):
        t = i * 0.0667
        if t > 1.0:
            r = scan(front=3.0)            # 회전 도중 앞이 열림
        pl, pr, st = roam.step(t, (0, 0, 0), B, r, None)
        seq.append((st, pl, pr))
    rev = [s for s in seq if s[1] < -5 and s[2] < -5]
    ok &= not rev
    print(f"  [{'OK ' if not rev else 'FAIL'}] 회전으로 풀리면 후진 0회 "
          f"(후진 명령 {len(rev)}개) · 전이 "
          f"{' → '.join(dict.fromkeys(s[0] for s in seq))}")
    return ok


if __name__ == "__main__":
    want_drive = "--unit" not in sys.argv
    want_unit = "--drive" not in sys.argv
    ok = True
    if want_drive:
        print("=== 주행 시나리오 (4x4 아레나, 오른쪽 아래 코너 시작) ===")
        ok &= run_drive()
    if want_unit:
        print("\n=== ESCAPE 회복 우선순위 (전진 → 회전 → 후진) ===")
        ok &= run_unit()
    print("\n결과:", "전부 통과" if ok else "실패 있음")
    sys.exit(0 if ok else 1)
