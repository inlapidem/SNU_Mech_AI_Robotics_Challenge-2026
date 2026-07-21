#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""포획 종단 commit 기하 A/B — sim 느슨값 vs field-verified 값.

동기 (2026-07-21 전략분석 워크플로우):
  mission_fsm 의 blind push 는 verify_range < blind_push_range(0.40) 또는 range=None
  에서 진입해 capture_push_max(0.62m) / capture_push_limit_s(11s) 까지 직진한다.
  반면 유일하게 실기검증된 컨트롤러 deployment/capture_demo.py 는 훨씬 보수적이다:
      · |bearing| <= 12deg AND range <= blind_enter(0.24m) 에서만 푸시
      · 기억추정만으로는 절대 푸시 안 함
      · 빗맞으면 0.35~0.70m 에서 재획득(APPROACH 복귀), push_timeout 6s -> MISS
  (capture_demo.py:184-188, 228-238, 316-326)

  채택된 0.62/11s 의 근거였던 '2026-07-17 검출미스 배터리(p=0.25/0.45)'는 shipped
  sim_mission.py 에 존재하지 않고, 현행 결정론 SimPerception 에서 이 파라미터는
  점수에 무효다(스윕 전 구간 +40 고정). 실측 recall 은 86~100% 라 전제와도 어긋난다.

여기서 재는 것: 종단 commit 을 field 기하(0.24/0.30/6s)로 좁히면 sim 점수를
  깎지 않으면서 -40 노출(blind push 중 이웃 동반포획)과 헛푸시 시간을 줄이는가.
  ★ mission_fsm.py 를 고치지 않는다 — params_override 로 효과만 측정한다.
  ★ OCCLUSION=True(현실적 가림)에서도 같이 돌린다 — 기본 False 는 물체 투명 모델.

사용:
  python3 navigation/sweep_capture_commit.py                 # 24시드, 가림 on/off
  python3 navigation/sweep_capture_commit.py --seeds 48
  python3 navigation/sweep_capture_commit.py --mode bothcube  # cube 공지 경기
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_mission import run_match, SimPerception, FRUITS   # noqa: E402

EFF = 0.73
CRUISE = 0.20                                # 실기 도달 가능 안전 상한 (우모터 88%)
BASE = dict(cruise_v=CRUISE, eff_speed=round(CRUISE * EFF, 4))

# 종단 commit 기하 — 상위 FSM(선택/메모리/veto)은 동일, 마지막 blind push 만 다름.
#  A  = 현행. B1 = '푸시 예산'만 축소(진입게이트 유지) → 시뮬이 판정 가능한 부분.
#  B2 = 필드 진입거리(0.24)까지 축소 → ⚠ 시뮬 VERIFY_BLIND=0.28 맹점 탓에 진입
#       자체가 불가(점수 붕괴). 이는 실기가 아니라 '시뮬의 한계'를 보여주는 대조군.
#       진입거리(0.24)의 실효는 deployment/capture_demo.py 로만 검증된다.
CONFIGS = [
    ("A_현행        (0.40/0.62/11s)",
     dict(blind_push_range=0.40, capture_push_max=0.62, capture_push_limit_s=11.0)),
    ("B1_예산축소   (0.40/0.35/6s)",
     dict(blind_push_range=0.40, capture_push_max=0.35, capture_push_limit_s=6.0)),
    ("B2_필드진입   (0.24/0.30/6s·시뮬한계 대조군)",
     dict(blind_push_range=0.24, capture_push_max=0.30, capture_push_limit_s=6.0)),
]

# 관심 이벤트(부분일치로 집계 — 정확한 이름 변화에 강건).
EVENT_KEYS = {
    "capture_enter":  "APPROACH->CAPTURE",
    "capture_miss":   "CAPTURE_MISSED",
    "uc_drop":        "UNINTENDED_CAPTURE",
    "approach_giveup": "APPROACH->RETREAT",
    "goto_giveup":    "GOTO->RETREAT",
}


def target_sets(mode, rng_choice):
    """--mode 에 따라 목표 강제. both=랜덤, set2=과일만, bothcube=cube 공지."""
    if mode == "both":
        return None                                      # run_match 가 랜덤 선택
    if mode == "set2":
        return {"set1": "__none__", "set2": rng_choice}  # set1 미지정 → 과일만 득점
    if mode == "bothcube":
        return {"set1": "cube", "set2": rng_choice}
    raise SystemExit(f"unknown mode {mode}")


def run_cfg(name, override, seeds, occ, mode):
    SimPerception.OCCLUSION = occ
    pts, good, bad, wall, spill = [], [], [], [], []
    ev = {k: [] for k in EVENT_KEYS}
    for s in range(seeds):
        # 시드별 과일 목표를 결정적으로 골라 페어드 비교 (A/B 가 같은 배치·목표)
        fruit = FRUITS[s % len(FRUITS)]
        tgt = target_sets(mode, fruit)
        p = dict(BASE); p.update(override)
        r = run_match(seed=s, targets=tgt, params_override=p)
        pts.append(r["points"]); good.append(r["good"]); bad.append(r["bad"])
        wall.append(r["wall_hits"]); spill.append(r["spilled"])
        counts = {k: 0 for k in EVENT_KEYS}
        for _, e in r["events"]:
            for k, sub in EVENT_KEYS.items():
                if sub in e:
                    counts[k] += 1
        for k in EVENT_KEYS:
            ev[k].append(counts[k])
    m = statistics.mean
    return dict(name=name, points=m(pts), good=m(good), bad=m(bad),
                wall=m(wall), spill=m(spill),
                ev={k: m(v) for k, v in ev.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--mode", default="both",
                    choices=["both", "set2", "bothcube"])
    args = ap.parse_args()

    for occ in (False, True):
        tag = "가림 ON (현실적)" if occ else "가림 OFF (투명·기존기준선)"
        print(f"\n=== 포획 commit A/B · mode={args.mode} · cruise={CRUISE} · "
              f"{args.seeds}시드 · {tag} ===")
        print(f"{'설정':<44}{'점수':>7}{'하역':>6}{'오픽업':>7}{'벽':>5}"
              f"{'스필':>6}{'포획진입':>8}{'빗맞':>6}{'UC배출':>7}{'접근포기':>8}")
        rows = [run_cfg(n, o, args.seeds, occ, args.mode) for n, o in CONFIGS]
        for r in rows:
            e = r["ev"]
            print(f"{r['name']:<44}{r['points']:>7.1f}{r['good']:>6.2f}"
                  f"{r['bad']:>7.2f}{r['wall']:>5.1f}{r['spill']:>6.2f}"
                  f"{e['capture_enter']:>8.2f}{e['capture_miss']:>6.2f}"
                  f"{e['uc_drop']:>7.2f}{e['approach_giveup']:>8.2f}")
        d = rows[1]["points"] - rows[0]["points"]
        db = rows[1]["bad"] - rows[0]["bad"]
        print(f"  Δ(B1−A): 점수 {d:+.1f}, 오픽업 {db:+.2f}  "
              f"→ 점수 중립(±노이즈)이면 푸시예산 축소는 무손실 = -40 보험으로 채택 가능")
        print(f"  ⚠ B2 붕괴는 시뮬 VERIFY_BLIND(0.28) 맹점 탓 — 진입거리 실효는 "
              f"capture_demo 실기로만 판정")


if __name__ == "__main__":
    main()
