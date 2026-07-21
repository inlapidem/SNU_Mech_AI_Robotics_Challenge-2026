#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""미션 시간예산·경로 시뮬 — 180초에 타깃 7개를 정말 옮길 수 있는가?

왜 이걸 먼저 하는가: mission_demo.py 의 설계는 '7개 전부'를 목표로 할지
'확실한 몇 개'만 노릴지, 그리고 탐색 경로를 어떤 모양으로 할지에 따라 완전히
달라진다. 구현한 뒤에 알면 되돌리기 어려우므로 숫자를 먼저 낸다.

★ 1차 시도의 교훈 (남겨둔다): 실패 모드 없이 돌렸더니 pwm 110/255, 정책 2종이
   **전부 7.00개 100% 달성**이 나왔다. 판별력 0. 이상적 조건에서 주행 20.8m /
   123초(pwm110)면 시간은 남는다는 뜻이고, 따라서 **이 미션의 병목은 시간이 아니라
   실행 신뢰도**다. 그래서 아래 실패 모드를 넣어야 비로소 설정 간 차이가 보인다:
     * 오분류 -> 비타깃을 담음 -> 2배 페널티 (점수를 깎는 유일한 경로)
     * 푸시 빗맞음 -> push_timeout 6s 낭비 후 재시도
     * 기억 위치 오차 -> 가서 보니 없음 -> 재탐색
     * 밀집 우회 -> 직선거리보다 실제 주행이 길다
   실패율은 전부 ASSUME 이다. 절대값이 아니라 **설정 간 상대 추세**로 읽을 것.

이 시뮬이 모델하지 않는 것: 라이다/점유격자/SLAM. 그건 explore_sim 의 몫이다.
여기서는 '이동 몇 초, 확정 몇 초, 푸시 몇 초'만 회계한다.

경기 조건 (2026-07-20 팀 확인):
  * 아레나 4x4 m. 시작 = 왼쪽 아래 코너, 반납 구역 = 왼쪽 위 코너 (태극기 스티커)
  * 물체 28개: 과일 4종 x 3 = 12, 모형 4종 x 4 = 16
  * 타깃 = 지정 과일 1종(3개) + 지정 모형 1종(4개) = 7개. 경기 전 통보
  * 비타깃 21개는 담으면 2배 페널티. 밀치는 것 자체는 무관
  * 빈 용량 2개. 제한시간 180초. 반납은 구역 안에 완전히 들어가야 인정

실측에서 가져온 상수 (가정은 ASSUME 표시):
  * HFOV 73.4/75.7deg -> +-37deg  : calib/front_{left,right}.json 실측
  * 스테레오 정밀 0.2~1.5m         : deployment/stereo_range.py
  * blind_enter .24 / hold_dist .38 / confirm_timeout 6 / push_timeout 6
                                   : deployment/capture_demo.py CaptureController
  * ShapeVoter need 3 (cube 5)     : deployment/capture_demo.py
  * 0.32 m/s(pwm110) 0.73(pwm255)  : sim/explore_sim.py 실측 (K_PWM 자체는 ASSUME)

사용:
  python3 sim/mission_budget.py                  # 경로안 x 속도 비교
  python3 sim/mission_budget.py --trials 400
  python3 sim/mission_budget.py --verbose        # 한 판 상세
  python3 sim/mission_budget.py --stress         # 실패율 민감도
"""
import argparse
import math
import random

# ------------------------------------------------------------------ 경기 상수
ARENA = 2.0
T_LIMIT = 180.0
BIN_CAP = 2
START = (-1.65, -1.65)
START_YAW = math.pi / 2
DEPOT = (-1.60, 1.60)
DEPOT_R = 0.45                      # ASSUME — 규칙 문서로 확인 필요
N_TARGET = 7
STARTUP_S = 6.0                     # 아레나 인식·기동 준비 ASSUME

FRUITS = ["apple", "banana", "orange", "pineapple"]
SHAPES = ["cube", "dodecahedron", "icosahedron", "octahedron"]

# ------------------------------------------------------------------ 지각 상수
HFOV_HALF = math.radians(37.0)      # 실측
R_DETECT = 3.0                      # ASSUME (8cm @ fx858 -> 3m 에서 23px)
R_CLASSIFY = 1.8                    # ASSUME
OBJ_R = 0.05
FPS = 8.0                           # ASSUME
VOTE_N, VOTE_N_CUBE = 3, 5

# ------------------------------------------------------------------ 구동 상수
SPEEDS = {110: 0.32, 255: 0.73}
OMEGA = {110: 1.4, 255: 2.6}
T_ACCEL = 0.35
DETOUR = 1.25                       # 밀집 회피로 실제 주행이 직선의 몇 배 ASSUME

D_PUSH, D_HOLD = 0.24, 0.38
T_PUSH, T_ALIGN, T_SETTLE = 0.9, 0.6, 0.3
T_PUSH_TIMEOUT = 6.0                # capture_demo push_timeout
T_DEPOSIT_PUSH, T_DEPOSIT_REV = 1.2, 1.0

# ------------------------------------------------------------------ 실패 모드
P_MISCLASS = 0.06                   # 확정했는데 실제로는 다른 종류 ASSUME
P_PUSH_MISS = 0.15                  # 블라인드 푸시 빗맞음 ASSUME
POS_ERR = 0.12                      # 기억 위치 오차 sigma [m] ASSUME
PENALTY = 2.0                       # 오포획 1개 = 정답 2개어치 감점


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def place_objects(rng, min_gap=0.30, margin=0.28):
    objs, tries = [], 0
    want = [(f, "fruit") for f in FRUITS for _ in range(3)]
    want += [(s, "shape") for s in SHAPES for _ in range(4)]
    rng.shuffle(want)
    for cls, kind in want:
        while tries < 40000:
            tries += 1
            x = rng.uniform(-ARENA + margin, ARENA - margin)
            y = rng.uniform(-ARENA + margin, ARENA - margin)
            if math.hypot(x - DEPOT[0], y - DEPOT[1]) < DEPOT_R + 0.25:
                continue
            if math.hypot(x - START[0], y - START[1]) < 0.40:
                continue
            if any(math.hypot(x - o["x"], y - o["y"]) < min_gap for o in objs):
                continue
            objs.append(dict(x=x, y=y, cls=cls, kind=kind, id=len(objs)))
            break
    return objs


def visible(rob, obj, others):
    dx, dy = obj["x"] - rob["x"], obj["y"] - rob["y"]
    d = math.hypot(dx, dy)
    if d > R_DETECT or d < 0.05:
        return False, False
    brg = wrap_pi(math.atan2(dy, dx) - rob["yaw"])
    if abs(brg) > HFOV_HALF:
        return False, False
    for o in others:
        if o["id"] == obj["id"]:
            continue
        ox, oy = o["x"] - rob["x"], o["y"] - rob["y"]
        od = math.hypot(ox, oy)
        if od >= d or od < 0.05:
            continue
        ob = wrap_pi(math.atan2(oy, ox) - rob["yaw"])
        if abs(wrap_pi(ob - brg)) < math.atan2(OBJ_R, od):
            return False, False
    return True, d <= R_CLASSIFY


# ------------------------------------------------------------------ 탐색 경로
def path_L_up_right():
    """ㄱ자 (위로 -> 오른쪽). 왼쪽 변을 타고 올라가 반납구역을 지나 위를 훑는다."""
    return [(-1.3, 0.0), (-1.3, 1.3), (1.3, 1.3), (1.3, 0.0)]


def path_L_right_up():
    """ㄱ자 (오른쪽 -> 위로). 아래를 훑고 오른쪽 변을 타고 올라간다."""
    return [(0.0, -1.3), (1.3, -1.3), (1.3, 1.3), (0.0, 1.3)]


def path_cross_center():
    """중앙 관통 ㄱ자 — 사용자 제안. 물체 밀도가 높은 한복판을 지난다."""
    return [(0.0, -0.6), (0.0, 0.9), (1.2, 0.9), (1.2, -0.6)]


def path_zigzag():
    """왕복 스윕 3줄 — 커버리지 최대, 주행거리도 최대."""
    return [(1.3, -1.2), (-1.3, -1.2), (-1.3, 0.0), (1.3, 0.0),
            (1.3, 1.2), (-1.3, 1.2)]


def path_perimeter():
    """외곽 순회 — 벽 근처 물체에 강하고 중앙에 약하다."""
    return [(1.4, -1.4), (1.4, 1.4), (-1.4, 1.4)]


PATHS = {
    "ㄱ자 위→오른쪽": path_L_up_right,
    "ㄱ자 오른쪽→위": path_L_right_up,
    "중앙관통 ㄱ자": path_cross_center,
    "지그재그 3줄": path_zigzag,
    "외곽 순회": path_perimeter,
}


class Sim:
    def __init__(self, rng, pwm=255, path="중앙관통 ㄱ자", spin_at_wp=True,
                 verbose=False, p_misclass=P_MISCLASS, p_miss=P_PUSH_MISS):
        self.rng = rng
        self.v, self.w = SPEEDS[pwm], OMEGA[pwm]
        self.objs = place_objects(rng)
        self.tf, self.ts = rng.choice(FRUITS), rng.choice(SHAPES)
        self.rob = dict(x=START[0], y=START[1], yaw=START_YAW)
        self.t = STARTUP_S
        self.bin, self.scored, self.wrong = [], 0, 0
        self.mem, self.votes, self.taken = {}, {}, set()
        self.waypoints = PATHS[path]()
        self.spin_at_wp = spin_at_wp
        self.verbose, self.log = verbose, []
        self.dist, self.n_deposit, self.n_miss, self.n_ghost = 0.0, 0, 0, 0
        self.p_misclass, self.p_miss = p_misclass, p_miss

    def is_target_cls(self, c):
        return c == self.tf or c == self.ts

    def say(self, m):
        if self.verbose:
            self.log.append(f"  t={self.t:6.1f}s  {m}")

    # ---- 지각 ----
    def observe(self, dt):
        n = max(1, int(dt * FPS))
        live = [o for o in self.objs if o["id"] not in self.taken]
        for o in live:
            vis, clsable = visible(self.rob, o, live)
            if not vis or not clsable:
                continue
            m = self.mem.setdefault(o["id"], dict(
                x=o["x"] + self.rng.gauss(0, POS_ERR),
                y=o["y"] + self.rng.gauss(0, POS_ERR), cls=None))
            need = VOTE_N_CUBE if o["cls"] == "cube" else VOTE_N
            self.votes[o["id"]] = self.votes.get(o["id"], 0) + n
            if self.votes[o["id"]] >= need and m["cls"] is None:
                if self.rng.random() < self.p_misclass:
                    # 오분류: 실제와 다른 라벨로 확정된다. 타깃으로 잘못 보면 담는다.
                    pool = [c for c in FRUITS + SHAPES if c != o["cls"]]
                    m["cls"] = self.rng.choice(pool)
                else:
                    m["cls"] = o["cls"]

    def advance(self, dt):
        self.t += dt
        self.observe(dt)

    def turn_to(self, yaw):
        d = abs(wrap_pi(yaw - self.rob["yaw"]))
        if d < 1e-3:
            return
        dt = d / self.w + T_ACCEL
        steps = max(1, int(d / 0.30))
        for i in range(steps):
            self.rob["yaw"] = wrap_pi(self.rob["yaw"]
                                      + wrap_pi(yaw - self.rob["yaw"]) / (steps - i))
            self.advance(dt / steps)
        self.rob["yaw"] = yaw

    def goto(self, x, y, stop_at=0.0):
        dx, dy = x - self.rob["x"], y - self.rob["y"]
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return
        self.turn_to(math.atan2(dy, dx))
        travel = max(0.0, d - stop_at)
        dt = travel * DETOUR / self.v + T_ACCEL        # 밀집 우회로 시간이 더 든다
        steps = max(1, int(travel / 0.25))
        for _ in range(steps):
            f = 1.0 / steps
            self.rob["x"] += dx / d * travel * f
            self.rob["y"] += dy / d * travel * f
            self.advance(dt / steps)
        self.dist += travel * DETOUR

    def out_of_time(self):
        return self.t >= T_LIMIT

    # ---- 행동 ----
    def spin(self, turns=8):
        for _ in range(turns):
            if self.out_of_time():
                return
            self.turn_to(wrap_pi(self.rob["yaw"] + 2 * math.pi / turns))

    def capture(self, oid):
        """접근 -> 확정 대기 -> 정렬 -> 푸시. 반환: 담았나."""
        m = self.mem.get(oid)
        if m is None:
            return False
        real = next((x for x in self.objs if x["id"] == oid), None)
        self.goto(m["x"], m["y"], stop_at=D_HOLD)
        if self.out_of_time():
            return False
        if real is None or real["id"] in self.taken:
            self.n_ghost += 1
            self.mem.pop(oid, None)
            return False
        # 기억 위치 오차로 실제와 어긋나면 재획득 비용
        off = math.hypot(self.rob["x"] - real["x"], self.rob["y"] - real["y"])
        if off > D_HOLD + 0.35:
            self.say(f"#{oid} 위치 어긋남 {off:.2f}m -> 재접근")
            self.advance(1.5)
            self.goto(real["x"], real["y"], stop_at=D_HOLD)
        if m["cls"] is None:
            waited = 0.0
            while m["cls"] is None and waited < 6.0 and not self.out_of_time():
                self.advance(0.25); waited += 0.25
            if m["cls"] is None:
                self.say(f"#{oid} 확정 실패 -> 포기 ({waited:.1f}s 낭비)")
                self.mem.pop(oid, None)
                return False
        if not self.is_target_cls(m["cls"]):
            self.say(f"#{oid} 비타깃({m['cls']}) -> 회피")
            return False
        self.advance(T_ALIGN)
        self.goto(real["x"], real["y"], stop_at=D_PUSH)
        if self.rng.random() < self.p_miss:
            self.n_miss += 1
            self.say(f"#{oid} 푸시 빗맞음 -> {T_PUSH_TIMEOUT}s 낭비 후 재시도")
            self.advance(T_PUSH_TIMEOUT)
            if self.out_of_time():
                return False
            self.advance(T_ALIGN)
        self.advance(T_PUSH + T_SETTLE)
        self.taken.add(oid)
        # 담긴 것의 '실제' 종류로 채점한다 — 오분류였다면 여기서 페널티가 확정된다
        self.bin.append(real["cls"])
        self.mem.pop(oid, None)
        tag = "OK" if self.is_target_cls(real["cls"]) else "★오포획"
        self.say(f"#{oid} {real['cls']} 담음 [{tag}] (빈 {len(self.bin)}/{BIN_CAP})")
        return True

    def deposit(self):
        if not self.bin:
            return
        self.say(f"반납 이동 ({len(self.bin)}개)")
        self.goto(DEPOT[0], DEPOT[1], stop_at=DEPOT_R * 0.4)
        if self.out_of_time():
            return
        self.advance(T_DEPOSIT_PUSH + T_DEPOSIT_REV)
        for c in self.bin:
            if self.is_target_cls(c):
                self.scored += 1
            else:
                self.wrong += 1
        self.n_deposit += 1
        self.say(f"반납 완료 (정답 {self.scored} · 오포획 {self.wrong})")
        self.bin = []

    # ---- 정책 ----
    def known_targets(self):
        return [(i, m) for i, m in self.mem.items()
                if i not in self.taken and m["cls"] is not None
                and self.is_target_cls(m["cls"])]

    def cost(self, m):
        d = math.hypot(m["x"] - self.rob["x"], m["y"] - self.rob["y"])
        turn = abs(wrap_pi(math.atan2(m["y"] - self.rob["y"],
                                      m["x"] - self.rob["x"]) - self.rob["yaw"]))
        return d / self.v + turn / self.w

    def grab_nearby(self, max_cost=4.0):
        """경로 주행 중 값싸게 딸 수 있는 타깃을 처리한다."""
        did = False
        while not self.out_of_time():
            if len(self.bin) >= BIN_CAP:
                self.deposit(); did = True; continue
            kt = [(i, m) for i, m in self.known_targets()
                  if self.cost(m) <= max_cost]
            if not kt:
                break
            oid, _ = min(kt, key=lambda p: self.cost(p[1]))
            self.capture(oid)
            did = True
        return did

    def run(self):
        # 1) 시작점에서 한 바퀴 — 좁은 FOV 를 메우는 첫 정보 획득
        self.spin()
        # 2) 탐색 경로를 따라가며 값싼 타깃을 딴다
        for wp in self.waypoints:
            if self.out_of_time():
                break
            self.goto(*wp)
            self.grab_nearby()
            if self.spin_at_wp and not self.out_of_time():
                self.spin(6)
                self.grab_nearby()
        # 3) 경로가 끝났는데 시간이 남으면 아는 타깃을 계속 회수
        stall = 0
        while not self.out_of_time() and self.scored + len(self.bin) < N_TARGET:
            if len(self.bin) >= BIN_CAP:
                self.deposit(); continue
            kt = self.known_targets()
            if kt:
                oid, _ = min(kt, key=lambda p: self.cost(p[1]))
                self.capture(oid); stall = 0; continue
            stall += 1
            if stall > 5:
                break
            self.goto(self.rng.uniform(-1.4, 1.4), self.rng.uniform(-1.4, 1.4))
            self.spin(6)
        if self.bin and not self.out_of_time():
            self.deposit()
        return dict(scored=self.scored, wrong=self.wrong,
                    net=self.scored - PENALTY * self.wrong,
                    t=min(self.t, T_LIMIT), dist=self.dist,
                    deposits=self.n_deposit, miss=self.n_miss, ghost=self.n_ghost,
                    stranded=len(self.bin))


def run_many(trials, **kw):
    res = [Sim(random.Random(2000 + i), **kw).run() for i in range(trials)]
    n = len(res)
    return dict(
        net=sum(r["net"] for r in res) / n,
        scored=sum(r["scored"] for r in res) / n,
        wrong=sum(r["wrong"] for r in res) / n,
        full=100.0 * sum(1 for r in res if r["scored"] >= N_TARGET) / n,
        dist=sum(r["dist"] for r in res) / n,
        t=sum(r["t"] for r in res) / n,
        dep=sum(r["deposits"] for r in res) / n,
        miss=sum(r["miss"] for r in res) / n,
        strand=sum(r["stranded"] for r in res) / n)


def table(trials):
    print(f"{'경로 x 속도':<28}{'순점수':>7}{'회수':>6}{'오포획':>7}"
          f"{'7개':>6}{'반납':>6}{'빗맞':>6}{'미반납':>7}{'주행m':>7}{'초':>6}")
    best = None
    for name in PATHS:
        for pwm in (110, 255):
            r = run_many(trials, pwm=pwm, path=name)
            print(f"{name + f' / {pwm}':<28}{r['net']:>7.2f}{r['scored']:>6.2f}"
                  f"{r['wrong']:>7.2f}{r['full']:>5.0f}%{r['dep']:>6.1f}"
                  f"{r['miss']:>6.1f}{r['strand']:>7.2f}{r['dist']:>7.1f}{r['t']:>6.0f}")
            if best is None or r["net"] > best[1]:
                best = (f"{name} / pwm {pwm}", r["net"])
    print(f"\n최고: {best[0]}  (순점수 {best[1]:.2f})")
    return best


def stress(trials, path, pwm):
    print(f"\n=== 실패율 민감도 ({path} / pwm {pwm}) ===")
    print(f"{'오분류율':>8}{'빗맞율':>8}{'순점수':>8}{'회수':>7}{'오포획':>8}{'7개':>6}")
    for pm in (0.0, 0.06, 0.12, 0.20):
        for pp in (0.05, 0.15, 0.30):
            r = run_many(trials, pwm=pwm, path=path, p_misclass=pm, p_miss=pp)
            print(f"{pm:>8.0%}{pp:>8.0%}{r['net']:>8.2f}{r['scored']:>7.2f}"
                  f"{r['wrong']:>8.2f}{r['full']:>5.0f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--path", default="중앙관통 ㄱ자")
    a = ap.parse_args()

    if a.verbose:
        s = Sim(random.Random(2001), pwm=255, path=a.path, verbose=True)
        r = s.run()
        print(f"타깃: {s.tf} + {s.ts}   경로: {a.path}")
        print("\n".join(s.log))
        print(f"\n결과: {r}")
        raise SystemExit(0)

    print(f"=== 미션 시간예산·경로 비교 (180s · 타깃 7 · 빈 2칸 · "
          f"오분류 {P_MISCLASS:.0%} · 빗맞 {P_PUSH_MISS:.0%}) ===\n")
    best = table(a.trials)
    if a.stress:
        stress(a.trials, a.path, 255)
