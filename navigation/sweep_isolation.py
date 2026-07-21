#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""타깃 선정에 '고립도'를 넣으면 헛걸음이 줄어드는가.

동기 (2026-07-20 이벤트 census, 24경기 가림 ON, 평균 42.1점):
    TOUR->GOTO              5.8/경기   타깃 선정
    GOTO->APPROACH          5.9/경기
      ├ APPROACH->CAPTURE   3.0/경기   포획 도달 (51%)
      └ APPROACH->RETREAT   2.8/경기   ★ 중도 포기 (48%)
    GOTO->RETREAT           1.9/경기   ★ 접근 전에 포기
    UNINTENDED_CAPTURE->DROP 2.0/경기  ★ 우발 포획
경기당 **4.7회를 헛걸음**한다. GOTO 가 시간의 33% 인데 그 절반가량이 버려진다.

원인은 물체 밀집이다. 규정상 50cm 격자 42지점에 28개(밀도 3.7개/m^2)가 놓이므로,
접근 중 verify 캠 시야에 **다른 물체가 더 가까이** 잡히면 FSM 이 오포획을 우려해
거부(VETO_OTHER)하고 재접근한다. 재시도 한도를 넘으면 그 이동이 통째로 버려진다.

그런데 `_select_target` (mission_fsm.py:1626) 은 점수를 `가치/왕복시간` 으로만
매긴다 — **주변이 한산한지는 전혀 안 본다.** 이웃이 많은 물체는 VETO 와 우발
포획을 부르는데도 같은 취급이다.

여기서 재는 것: 후보의 이웃 수로 점수를 깎으면 헛걸음이 줄어 점수가 오르는가.
  score *= 1 / (1 + K * n_neighbors)
  n_neighbors = 기억 속 물체 중 후보로부터 R 이내 (자기 자신 제외)

★ mission_fsm.py 를 고치지 않는다. 여기서는 monkeypatch 로 효과만 측정한다.
  이득이 확인되면 그때 본체에 반영할지 결정한다 (본체는 검증된 산출물이다).

사용:
  python3 navigation/sweep_isolation.py
  python3 navigation/sweep_isolation.py --seeds 32
"""
import argparse
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mission_fsm as MF                                   # noqa: E402
from sim_mission import run_match, SimPerception           # noqa: E402

EFF = 0.73
BASE = dict(cruise_v=0.21, eff_speed=round(0.21 * EFF, 4), approach_v=0.13)

_orig_select = MF.MissionFSM._select_target
ISO = {"k": 0.0, "r": 0.6}          # 전역 설정 (실험마다 갱신)


def _patched_select(self, t, pose):
    """원본을 그대로 부르되, 고립도 가중을 넣기 위해 _value 를 일시 감쌈.

    _select_target 은 score = _value(o)/trip 형태라, _value 를 후보별로
    이웃 수만큼 깎으면 선정 순위에 고립도가 반영된다. 원본 로직(게이트·컷오프)은
    건드리지 않으므로 부작용이 없다.
    """
    k, R = ISO["k"], ISO["r"]
    if k <= 0.0:
        return _orig_select(self, t, pose)
    objs = self.memory.objects
    orig_value = self.__class__._value

    def value_iso(inner_self, o):
        v = orig_value(inner_self, o)
        n = 0
        for q in objs:
            if q is o or q["status"] not in ("open", "defer"):
                continue
            if math.hypot(q["x"] - o["x"], q["y"] - o["y"]) <= R:
                n += 1
        return v / (1.0 + k * n)

    self.__class__._value = value_iso
    try:
        return _orig_select(self, t, pose)
    finally:
        self.__class__._value = orig_value


MF.MissionFSM._select_target = _patched_select


def run_cfg(params, seeds, k=0.0, r=0.6):
    ISO["k"], ISO["r"] = k, r
    rows = [run_match(seed=s, params_override=params) for s in range(seeds)]
    pts = [x["points"] for x in rows]
    n = len(rows)
    sd = statistics.stdev(pts) if n > 1 else 0.0
    ev = {}
    for x in rows:
        for _, e in x["events"]:
            key = e.split("(")[0]
            ev[key] = ev.get(key, 0) + 1
    return dict(mean=statistics.mean(pts), se=sd / (n ** 0.5), median=statistics.median(pts),
                worst=min(pts), good=statistics.mean([x["good"] for x in rows]),
                bad=sum(x["bad"] for x in rows), wall=sum(x["wall_hits"] for x in rows),
                retreat=(ev.get("APPROACH->RETREAT", 0) + ev.get("GOTO->RETREAT", 0)) / n,
                unintended=ev.get("UNINTENDED_CAPTURE->DROP", 0) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    a = ap.parse_args()
    SimPerception.OCCLUSION = True

    print("=" * 96)
    print(f"  타깃 선정 정책 + 고립도 가중  (시드 {a.seeds}, 가림 ON, cruise 0.21/appr 0.13)")
    print("=" * 96)
    print(f"{'설정':<32}{'평균':>7}{'표준오차':>8}{'중앙':>6}{'최악':>6}"
          f"{'하역':>6}{'오픽업':>7}{'벽':>5}{'헛걸음':>8}{'우발포획':>9}{'Δ':>7}")

    trials = [("기준 value_time", dict(BASE), 0.0, 0.6)]
    # 기존 정책들 — 이미 구현돼 있으니 먼저 확인
    for pol in ("nearest", "value_first", "pair_aware"):
        trials.append((f"정책 {pol}", dict(BASE, target_policy=pol), 0.0, 0.6))
    # 고립도 가중 (신규 제안)
    for k in (0.15, 0.30, 0.50):
        trials.append((f"고립도 K={k} R=0.6", dict(BASE), k, 0.6))
    for r in (0.45, 0.80):
        trials.append((f"고립도 K=0.30 R={r}", dict(BASE), 0.30, r))

    base = None
    rows = []
    for label, params, k, r in trials:
        res = run_cfg(params, a.seeds, k, r)
        if base is None:
            base = res["mean"]
        d = res["mean"] - base
        rows.append((label, res))
        print(f"{label:<32}{res['mean']:>7.1f}{res['se']:>8.1f}{res['median']:>6.0f}"
              f"{res['worst']:>6.0f}{res['good']:>6.2f}{res['bad']:>7d}{res['wall']:>5d}"
              f"{res['retreat']:>8.2f}{res['unintended']:>9.2f}{d:>+7.1f}")

    b0 = rows[0][1]
    win = [(l, r) for l, r in rows[1:]
           if r["mean"] - base > 2 * (r["se"] + b0["se"]) and r["bad"] <= b0["bad"]]
    if win:
        w = max(win, key=lambda kv: kv[1]["mean"])
        print(f"\n  => 채택 후보: {w[0]}  ({w[1]['mean']:.1f}, {w[1]['mean']-base:+.1f})")
        print(f"     헛걸음 {b0['retreat']:.2f} -> {w[1]['retreat']:.2f} / 경기")
    else:
        print("\n  => 통계적으로 유의한 개선 없음")
    print("\n주의: 평균 차이가 표준오차 합의 2배 미만이면 노이즈다.")


if __name__ == "__main__":
    main()
