#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""구동계 동특성 실측 — 시뮬이 '가정'으로 쓰고 있는 상수 4개를 한 번에 잰다.

왜 필요한가: 미션 시간예산(180초에 7개)의 답이 주행속도 하나에 달려 있는데
지금 저장소에 서로 다른 두 추정치가 있고 **둘 다 실측이 아니다**:
  * motor_control/params.yaml  max_wheel_speed 0.2 m/s  (기어비 131 + 45~76RPM 역산)
  * sim/explore_sim.py         K_PWM -> PWM255 에서 0.73 m/s (근거 없는 가정)
3.6배 차이다. 0.2 가 맞으면 반납 왕복 4회에만 140초가 들어 7개는 불가능하고,
0.73 이면 여유가 있다. 이 값 하나로 미션 설계 전체가 갈린다.

재는 것 (전부 엔코더 기반. 줄자는 선택):
  1. 정지마찰 하한   바퀴가 실제로 돌기 시작하는 최소 PWM  -> min_pwm
  2. PWM->속도 곡선  여러 PWM 에서 정상상태 속도            -> K_PWM (선형인지도 확인)
  3. 구동계 시정수   1차 지연 tau -> 제동거리 ~ v*tau        -> sim TAU
  4. 제자리 각속도   피벗 회전 rad/s                          -> sim OMEGA
  5. 타행 거리       정지 명령 후 더 굴러간 거리              -> 감속 램프 설계

펌웨어 규약 (firmware/motor_fw/motor_fw.ino 실측 확인):
  * "M <l> <r>\\n"  -255..255, 부호가 방향. 300ms 무명령이면 자동 정지
    -> 그래서 이 스크립트는 40ms 마다 명령을 재전송한다
  * "E <l> <r> <ir>\\n" 20ms 주기. **양쪽 다 전진 = 틱 증가** (오른쪽은 ISR 부호반전)
  * ENC A상 RISING 만 카운트 -> ticks_per_rev = 11 PPR x 131 기어비 = 1441 (공칭)

정확도 (가짜 아두이노로 진값을 심어 검증, 2026-07-20):
  진값 0.45 m/s / tau 0.30 / min_pwm 35 를 각각 0.454 / 0.339 / 35 로 복원.
  * 속도는 오차 1% 이내 — 지수적합이라 주행이 짧아도 편향이 없다
  * tau 는 +13% 정도 높게 나온다. 틱값이 최대 20ms(샘플주기의 절반) 낡은 채로
    샘플링되어 그만큼 지연이 더해지는 계통 편향이다. 제동거리를 보수적으로
    보게 되는 방향이라 안전측이지만, 값을 쓸 때 이 편향을 감안할 것.
  * 각속도도 같은 이유로 소폭 높게 나온다 (+8%).

안전:
  * 공간이 필요 없는 시험(피벗·정지마찰)부터 하고 직진은 나중에 한다
  * 매 직진 시험 전에 사용자 확인을 받는다
  * Ctrl-C / 예외 / 정상종료 어느 경로로도 반드시 모터를 세운다
  * 기본 주행시간 1.2초 — 최악(0.73m/s)에도 타행 포함 약 1m

사용 (모터 구동 전원이 켜져 있어야 바퀴가 돈다):
  python3 motor_control/measure_dynamics.py              # 전체 (권장)
  python3 motor_control/measure_dynamics.py --spin-only  # 공간 없을 때: 피벗만
  python3 motor_control/measure_dynamics.py --secs 0.8   # 공간이 좁으면 짧게
  python3 motor_control/measure_dynamics.py --pwms 110,255

결과는 화면 요약 + JSON 원자료(--out) 로 남는다. 원자료가 있어야 나중에
시정수 재적합이나 다른 모델 검토를 로봇 없이 할 수 있다.

★ motor_bridge 가 떠 있으면 /dev/ttyACM0 를 잡고 있어 실패한다. 먼저 끌 것.
"""
import argparse
import json
import math
import os
import sys
import threading
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial 이 필요합니다:  pip install pyserial")

DEFAULT_PARAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.yaml")
CMD_PERIOD = 0.04          # 워치독 300ms 대비 충분히 잦게 재전송
SETTLE_FRAC = 0.45         # 주행 후반 이 비율 구간을 '정상상태'로 본다


def load_params(path):
    v = {"wheel_radius": 0.033, "ticks_per_rev": 1441.0, "wheel_base": 0.36,
         "port": "/dev/ttyACM0", "baud": 115200}
    try:
        import yaml
        with open(path) as f:
            p = yaml.safe_load(f)["/motor_bridge"]["ros__parameters"]
        for k in v:
            if k in p:
                v[k] = p[k]
    except Exception as e:
        print(f"[i] params.yaml 로드 생략 ({e}) — 기본값 사용")
    return v


class Board:
    """시리얼 I/O. 리더는 별도 스레드, 명령은 호출자가 주기적으로 재전송."""

    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.lock = threading.Lock()
        self.ticks = (0, 0)
        self.ir = 1
        self.n = 0
        self.samples = []          # (t, l, r, ir) — 원자료
        self.recording = False
        self._stop = False
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        buf = b""
        while not self._stop:
            try:
                data = self.ser.read(256)
            except serial.SerialException:
                break
            if not data:
                continue
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                p = line.decode(errors="ignore").strip().split()
                if len(p) >= 3 and p[0] == "E":
                    try:
                        l, r = int(p[1]), int(p[2])
                        ir = int(p[3]) if len(p) >= 4 else 1
                    except ValueError:
                        continue
                    now = time.time()
                    with self.lock:
                        self.ticks = (l, r)
                        self.ir = ir
                        self.n += 1
                        if self.recording:
                            self.samples.append((now, l, r, ir))

    def send(self, pl, pr):
        try:
            self.ser.write(f"M {int(pl)} {int(pr)}\n".encode())
        except serial.SerialException:
            pass

    def get(self):
        with self.lock:
            return self.ticks

    def run_pwm(self, pl, pr, secs, coast_s=0.7):
        """pl/pr 을 secs 동안 유지하며 (시각, 좌틱, 우틱) 을 직접 샘플링한다.

        ★ 타임스탬프를 리더 스레드가 아니라 **이 루프**가 찍는 이유: OS 시리얼은
        데이터를 뭉쳐서 준다(read(256) 가 20ms 짜리 줄 10개를 한 번에 반환).
        파싱 시각으로 찍으면 서로 다른 시점의 샘플이 같은 시각을 갖게 되어
        인접 차분 속도가 무의미해진다(실제로 이 버그로 tau 적합이 전부 실패했다).
        여기서는 루프가 자기 시계로 등간격 샘플링하므로 틱값이 최대 20ms 낡을 뿐
        **시각은 정확**하다. 속도 추정에는 이쪽이 훨씬 낫다.

        반환: (series, t0, t_cmd_end)  — series = [(t, l, r), ...]
        """
        series = []
        t0 = time.time()
        try:
            while True:
                now = time.time()
                if now - t0 >= secs:
                    break
                self.send(pl, pr)
                l, r = self.get()
                series.append((now, l, r))
                time.sleep(CMD_PERIOD)
        finally:
            self.stop()
        t_cmd_end = time.time()
        t1 = t_cmd_end
        while time.time() - t1 < coast_s:          # 타행 구간도 같은 방식으로 기록
            now = time.time()
            l, r = self.get()
            series.append((now, l, r))
            time.sleep(CMD_PERIOD)
        return series, t0, t_cmd_end

    def stop(self):
        for _ in range(4):
            self.send(0, 0)
            time.sleep(0.02)

    def close(self):
        self.stop()
        self._stop = True
        self._t.join(timeout=1.0)
        try:
            self.ser.close()
        except Exception:
            pass


# ------------------------------------------------------------------ 분석
def velocity_series(series, m_per_tick, t0, t_cmd_end):
    """(t,l,r) -> [(t_rel, v_left, v_right)] [m/s]. 구동 명령 구간만."""
    out = []
    for i in range(1, len(series)):
        ta, la, ra = series[i - 1]
        tb, lb, rb = series[i]
        if tb > t_cmd_end:
            break
        dt = tb - ta
        if dt <= 1e-4:
            continue
        out.append((tb - t0, (lb - la) * m_per_tick / dt,
                    (rb - ra) * m_per_tick / dt))
    return out


def fit_first_order(vs, tau_lo=0.03, tau_hi=2.0, n_grid=240):
    """v(t) = v_ss * (1 - exp(-t/tau)) 를 (v_ss, tau) 동시 적합.

    왜 평균이 아니라 적합인가: 주행이 짧으면(secs ~ 4*tau 미만) 후반 평균조차
    아직 v_ss 에 도달하지 못해 **속도를 체계적으로 과소평가**한다. 실측에서
    진값 0.45 를 0.415 로(-8%) 낮게 읽은 원인이 이것이었다. 적합은 곡선의
    점근값을 직접 추정하므로 짧은 주행에서도 편향이 없다.

    방법: tau 를 격자탐색하고, 각 tau 에서 v_ss 는 최소제곱 해석해로 바로 구한다
    (기저 b(t)=1-exp(-t/tau) 에 대해 v_ss = <b,v>/<b,b>). 잔차 최소 조합 채택.
    반환 (v_ss, tau, rms) 또는 None.
    """
    pts = [(t, (a + b) / 2.0) for t, a, b in vs if t > 1e-6]
    if len(pts) < 6:
        return None
    best = None
    for i in range(n_grid):
        tau = tau_lo * (tau_hi / tau_lo) ** (i / (n_grid - 1.0))   # 로그 격자
        sbb = sbv = 0.0
        for t, v in pts:
            b = 1.0 - math.exp(-t / tau)
            sbb += b * b
            sbv += b * v
        if sbb < 1e-12:
            continue
        v_ss = sbv / sbb
        rss = 0.0
        for t, v in pts:
            e = v - v_ss * (1.0 - math.exp(-t / tau))
            rss += e * e
        rms = math.sqrt(rss / len(pts))
        if best is None or rms < best[2]:
            best = (v_ss, tau, rms)
    return best


def tail_mean(vs, frac=SETTLE_FRAC):
    """후반 구간 평균 — 적합의 교차검증용(적합이 이상하면 여기서 드러난다)."""
    if not vs:
        return 0.0, 0.0
    t_end = vs[-1][0]
    lo = t_end * (1.0 - frac)
    sel = [v for v in vs if v[0] >= lo]
    if not sel:
        return 0.0, 0.0
    return (sum(s[1] for s in sel) / len(sel),
            sum(s[2] for s in sel) / len(sel))


def coast(series, m_per_tick, t_cmd_end):
    """정지 명령 이후 추가로 굴러간 거리 [m]."""
    after = [s for s in series if s[0] >= t_cmd_end]
    if len(after) < 2:
        return 0.0
    return (abs(after[-1][1] - after[0][1])
            + abs(after[-1][2] - after[0][2])) / 2.0 * m_per_tick


def ask(msg):
    try:
        return input(msg)
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\n사용자 중단")


# ------------------------------------------------------------------ 시험
def test_stiction(bd, m_per_tick, res):
    print("\n[1/5] 정지마찰 하한 — PWM 을 조금씩 올려 바퀴가 도는 순간을 찾는다.")
    print("      로봇이 아주 천천히 조금 움직입니다 (수 cm).")
    ask("      준비되면 Enter (건너뛰려면 Ctrl-C)... ")
    found = None
    for pwm in range(15, 121, 5):
        base = bd.get()
        t0 = time.time()
        while time.time() - t0 < 0.45:
            bd.send(pwm, pwm)
            time.sleep(CMD_PERIOD)
        bd.stop()
        time.sleep(0.15)
        now = bd.get()
        moved = max(abs(now[0] - base[0]), abs(now[1] - base[1]))
        dist = moved * m_per_tick
        print(f"      PWM {pwm:3d} -> {moved:5d} tick ({dist*100:5.1f} cm)")
        if moved > 30:                     # 노이즈/떨림이 아닌 확실한 회전
            found = pwm
            break
    res["min_pwm"] = found
    print(f"      => 정지마찰 하한 ~ PWM {found}" if found
          else "      => 120 까지 안 움직임 — 구동 전원/배선 확인 필요")


def test_pivot(bd, m_per_tick, wheel_base, pwms, secs, res):
    print("\n[2/5] 제자리 회전 각속도 — 공간이 거의 필요 없습니다 (그 자리에서 돕니다).")
    ask("      준비되면 Enter... ")
    out = {}
    for pwm in pwms:
        series, t0, t_end = bd.run_pwm(-pwm, pwm, secs)   # 좌 후진/우 전진 = 좌회전
        vs = velocity_series(series, m_per_tick, t0, t_end)
        if len(vs) < 6:
            print(f"      PWM {pwm}: 샘플 부족 ({len(vs)}) — 텔레메트리 확인")
            continue
        vl, vr = tail_mean(vs)
        # 피벗은 좌우 부호가 반대라 평균이 0 이 된다 -> 회전속도로 바꿔 적합한다
        spin = [(t, (b - a) / 2.0, (b - a) / 2.0) for t, a, b in vs]
        fit = fit_first_order(spin)
        v_spin = fit[0] if fit else (vr - vl) / 2.0
        omega = 2.0 * v_spin / wheel_base
        out[pwm] = dict(v_left=vl, v_right=vr, omega=omega,
                        tau=fit[1] if fit else None,
                        omega_tail=(vr - vl) / wheel_base)
        print(f"      PWM {pwm:3d} -> 좌 {vl:+.3f} 우 {vr:+.3f} m/s · "
              f"각속도 {omega:.2f} rad/s ({math.degrees(omega):.0f} deg/s)")
        time.sleep(0.4)
    res["pivot"] = out


def test_straight(bd, m_per_tick, pwms, secs, res):
    print(f"\n[3/5] 직진 속도·시정수 — PWM 별로 {secs}초씩 전진합니다.")
    print("      ★ 앞쪽에 2m 이상 빈 공간이 필요합니다. 매 회 위치를 되돌려 주세요.")
    out = {}
    for pwm in pwms:
        ask(f"      PWM {pwm} 시험 준비되면 Enter... ")
        series, t0, t_end = bd.run_pwm(pwm, pwm, secs)
        vs = velocity_series(series, m_per_tick, t0, t_end)
        if len(vs) < 6:
            print(f"      PWM {pwm}: 샘플 부족 ({len(vs)}) — 텔레메트리 확인")
            continue
        vl_t, vr_t = tail_mean(vs)
        v_tail = (vl_t + vr_t) / 2.0
        fit = fit_first_order(vs)
        v = fit[0] if fit else v_tail
        tau = fit[1] if fit else None
        cst = coast(series, m_per_tick, t_end)
        skew = (vr_t - vl_t) / v_tail * 100.0 if abs(v_tail) > 1e-3 else 0.0
        out[pwm] = dict(v=v, v_tail=v_tail, v_left=vl_t, v_right=vr_t,
                        tau=tau, rms=fit[2] if fit else None,
                        coast=cst, skew_pct=skew)
        tau_s = f"tau {tau:.3f}s" if tau else "tau 적합실패"
        print(f"      PWM {pwm:3d} -> {v:.3f} m/s (적합)  {v_tail:.3f} (후반평균)  "
              f"{tau_s}")
        print(f"              좌 {vl_t:.3f} / 우 {vr_t:.3f} · 좌우차 {skew:+.1f}% · "
              f"타행 {cst*100:.1f} cm")
    res["straight"] = out


def test_ticks_check(bd, m_per_tick, res):
    print("\n[4/5] ticks_per_rev 교차검증 (선택) — 줄자로 실제 이동거리를 재서 입력.")
    a = ask("      할까요? [y/N] ").strip().lower()
    if a != "y":
        res["tick_check"] = None
        return
    base = bd.get()
    ask("      시작 위치를 표시하고 Enter (PWM 150 으로 1.5초 전진)... ")
    bd.run_pwm(150, 150, 1.5)
    now = bd.get()
    ticks = (abs(now[0] - base[0]) + abs(now[1] - base[1])) / 2.0
    est = ticks * m_per_tick
    s = ask(f"      엔코더 기준 {est*100:.1f} cm 이동. 실제 거리 [cm] 입력: ").strip()
    try:
        real = float(s) / 100.0
    except ValueError:
        res["tick_check"] = None
        return
    ratio = est / real if real > 1e-6 else float("nan")
    res["tick_check"] = dict(ticks=ticks, est_m=est, real_m=real, ratio=ratio)
    print(f"      => 엔코더/실제 = {ratio:.3f}  "
          f"(1.0 이면 ticks_per_rev 정확. 벗어나면 현재값 x {ratio:.3f} 로 보정)")


def summarize(res, params):
    print("\n" + "=" * 66)
    print("측정 요약")
    print("=" * 66)
    st = res.get("straight") or {}
    if st:
        top = max(st.items(), key=lambda kv: kv[1]["v"])
        print(f"  최고속 (PWM {top[0]}) : {top[1]['v']:.3f} m/s")
        print(f"  params.yaml 현재값   : {params.get('max_wheel_speed', 0.2)} m/s")
        print(f"  sim/explore_sim 가정 : 0.73 m/s (PWM 255 환산)")
        vs = [(p, d["v"]) for p, d in sorted(st.items())]
        if len(vs) >= 2:
            k = [v / p for p, v in vs if p > 0]
            print(f"  PWM->속도 계수 K     : {min(k):.5f} ~ {max(k):.5f} m/s per PWM")
            print(f"    (편차가 크면 비선형 — 시뮬의 선형 K_PWM 가정을 고쳐야 함)")
        taus = [d["tau"] for d in st.values() if d.get("tau")]
        if taus:
            print(f"  시정수 tau           : {sum(taus)/len(taus):.3f} s "
                  f"(시뮬 가정 0.25)")
        cs = [d["coast"] for d in st.values()]
        if cs:
            print(f"  타행거리             : {min(cs)*100:.1f} ~ {max(cs)*100:.1f} cm")
        sk = [abs(d["skew_pct"]) for d in st.values()]
        if sk:
            print(f"  좌우 모터 편차       : 최대 {max(sk):.1f}%  "
                  f"(크면 개루프 직진이 휜다)")
    pv = res.get("pivot") or {}
    if pv:
        top = max(pv.items(), key=lambda kv: abs(kv[1]["omega"]))
        print(f"  최대 각속도 (PWM {top[0]}) : {abs(top[1]['omega']):.2f} rad/s "
              f"(시뮬 가정 2.6)")
    if res.get("min_pwm"):
        print(f"  정지마찰 하한        : PWM {res['min_pwm']} "
              f"(params.yaml min_pwm 22)")
    tc = res.get("tick_check")
    if tc and tc.get("ratio"):
        print(f"  ticks_per_rev 배율   : x{tc['ratio']:.3f}")
    print("=" * 66)

    v = None
    if st:
        v = max(d["v"] for d in st.values())
    if v:
        # 반납 왕복 3.5m 왕복 x 4회 = 28m 주행분
        t_dep = 28.0 / v
        print(f"\n시간예산 즉석 판정 (최고속 {v:.2f} m/s 기준):")
        print(f"  반납 왕복 4회(약 28m) 주행에만 {t_dep:.0f}초 / 180초")
        if t_dep > 120:
            print("  => 7개는 불가능. 목표 개수를 줄이거나 반납 전략을 바꿔야 한다.")
        elif t_dep > 80:
            print("  => 7개는 매우 빠듯. 포획당 여유가 10초 미만.")
        else:
            print("  => 7개 시도할 만하다. 포획 신뢰도가 병목.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--port", default=None)
    ap.add_argument("--pwms", default="80,140,200,255",
                    help="시험할 PWM 목록 (쉼표)")
    ap.add_argument("--secs", type=float, default=1.2, help="직진 시험 시간 [s]")
    ap.add_argument("--spin-only", action="store_true",
                    help="공간이 없을 때 — 피벗/정지마찰만")
    ap.add_argument("--out", default="runtime_logs/motor_dynamics.json")
    a = ap.parse_args()

    p = load_params(a.params)
    port = a.port or p["port"]
    m_per_tick = 2 * math.pi * p["wheel_radius"] / p["ticks_per_rev"]
    pwms = [int(x) for x in a.pwms.split(",") if x.strip()]

    print("=" * 66)
    print("구동계 동특성 실측")
    print("=" * 66)
    print(f"  포트 {port} · 바퀴반경 {p['wheel_radius']} m · "
          f"ticks_per_rev {p['ticks_per_rev']}")
    print(f"  1 tick = {m_per_tick*1000:.4f} mm · 트랙폭 {p['wheel_base']} m")
    print("  ★ 모터 구동 전원이 켜져 있어야 합니다.")
    print("  ★ motor_bridge 가 떠 있으면 포트 충돌로 실패합니다.")

    try:
        bd = Board(port, p["baud"])
    except serial.SerialException as e:
        sys.exit(f"\n시리얼 열기 실패: {e}\n  motor_bridge 가 떠 있는지 확인하세요.")

    res = dict(params=p, m_per_tick=m_per_tick, pwms=pwms, secs=a.secs)
    try:
        print("\n텔레메트리 대기...")
        t0 = time.time()
        while bd.n < 5 and time.time() - t0 < 5.0:
            time.sleep(0.1)
        if bd.n < 5:
            sys.exit("텔레메트리('E' 라인) 수신 없음 — 아두이노 연결/펌웨어 확인")
        print(f"  OK ({bd.n} 샘플 수신, IR raw={bd.ir})")

        test_stiction(bd, m_per_tick, res)
        test_pivot(bd, m_per_tick, p["wheel_base"], pwms, min(a.secs, 1.0), res)
        if not a.spin_only:
            test_straight(bd, m_per_tick, pwms, a.secs, res)
            test_ticks_check(bd, m_per_tick, res)
        summarize(res, p)
    except KeyboardInterrupt:
        print("\n중단됨 — 모터 정지")
    finally:
        bd.close()
        try:
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            with open(a.out, "w") as f:
                json.dump(res, f, indent=1, ensure_ascii=False, default=str)
            print(f"\n원자료 저장: {a.out}")
        except Exception as e:
            print(f"\n저장 실패: {e}")


if __name__ == "__main__":
    main()
