#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""최고속(PWM 255) 자율 탐사 — explore_demo 엔진에 고속 프로파일만 얹은 진입점.

왜 별도 파일인가: explore_demo.py 의 기본값(pwm_base 110)은 아레나에서 실제로
검증된 설정이라 그대로 보존한다. 최고속은 아직 실기 검증 전이므로 진입점을
분리해 '무엇을 돌렸는지'가 파일 이름으로 남게 한다. 엔진(주행·회피·아레나 인식·
스윕)은 explore_demo 를 그대로 쓰므로 버그 수정은 한 곳만 고치면 양쪽에 적용된다.

고속 프로파일이 바꾸는 것은 인자 4개뿐이다:
  --pwm-base 255    개활지 최고속 (아두이노 analogWrite 8비트 상한)
  --pwm-slow 55     장애물 근처 최저속. 절대값이라 base 를 올려도 안 따라 오른다.
                    정지마찰 극복 하한 22(motor_control/motor_bridge.py)의 2.5배,
                    capture_demo 의 scan_pwm 65 와 같은 급이라 확실히 구른다.
  --ramp-span 1.00  front_stop(0.55m) 부터 1.0m 에 걸쳐 55 → 255 로 가속.
                    즉 전방이 1.55m 이상 열려야 최고속이 나온다.
  --startup-pwm 130 시작 구간(아레나 락 전)만 속도 상한. 경기장 시작 지점에
                    3mm 단차가 있는데, 최고속으로 넘으면 충격이 커 스캔정합이
                    흔들리고 튈 수 있다. 넘어갈 때만 낮추고 이후 최고속으로 간다.
                    (단차를 못 넘으면 올리고, 너무 튀면 내린다.)

엔진 쪽이 이 값들을 받아 자동으로 따라가는 것 (explore_demo 참조):
  * Roam.fit_pwm    조향 시 (base-d, base+d) 가 255 를 넘으면 **비율을 유지한 채**
                    양쪽을 줄인다. 그냥 자르면 좌/우 비가 깨져 회전반경이
                    0.85m → 1.55m 로 벌어진다 (4m 아레나에서 치명적).
  * speed_time_scale  스캔 신선도·워치독·목표 재계산 주기를 속도에 반비례로 줄여
                    '시간'이 아니라 '이동거리' 예산을 보존한다.

★ 실기 검증 상태 (2026-07-20 기준)
  저속본(explore_demo, pwm_base 110)은 실기에서 확인됨 — 시작 시 벽으로 돌진하던
  문제 해결, 아레나 박스 정상 생성. **최고속본은 아직 실기 미검증이다.**
  시뮬(관성 포함, 빈 아레나)에서 4개 시나리오 벽 접촉 0회, 0.74 m/s 도달,
  스윕 86~92%. 다만 그 시뮬은 아레나 내부가 비어 있어 고속의 위험을 충분히
  재현하지 못한다. 또 PWM→속도 계수와 구동계 시정수가 **실측이 아닌 가정**이라
  절대 안전 여유는 미검증이다. 반드시 --dry-run 으로 먼저 확인하고,
  실주행은 --max-secs 로 짧게 끊어서 시작할 것.

  알려진 미해결 위험 (코드로 못 막는 것):
  * 경기 물체(약 8cm)는 라이다 장착 높이에서 안 보일 수 있다 → 속도가 곧
    미검출 충돌 에너지가 된다. 카메라 검출에 의존하는 구간이다.
  * 좌우 모터 특성차 보정이 없다 (완전 개루프) → 직진 편향이 속도에 비례해 커진다.
  * 장애물이 있는 아레나에서는 시뮬 기준 접촉이 잦다(저속에서도). 물체를 놓고
    돌릴 때는 저속본으로 먼저 확인할 것.

사용:
  python3 deployment/explore_fast.py --dry-run        # 모터 없이 먼저 확인
  python3 deployment/explore_fast.py --max-secs 60    # 짧게 실주행
  python3 deployment/explore_fast.py --pwm-base 180   # 뒤에 준 값이 이긴다
  # 저속 검증본은 python3 deployment/explore_demo.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from deployment.explore_demo import main   # noqa: E402

# 고속 프로파일. 사용자가 같은 옵션을 주면 뒤에 오므로 argparse 가 그쪽을 채택한다.
FAST_PROFILE = ["--pwm-base", "255", "--pwm-slow", "55", "--ramp-span", "1.00",
                "--startup-pwm", "130"]


if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + FAST_PROFILE + sys.argv[1:]
    print("[fast] 최고속 프로파일 " + " ".join(FAST_PROFILE)
          + "  (저속 검증본: deployment/explore_demo.py)")
    main()
