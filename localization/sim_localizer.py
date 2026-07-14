#!/usr/bin/env python3
# WSL 모의 검증: 4x4 경기장에서 wall_localizer_core 정확도/강건성 테스트 (ROS 불필요)
#
# 실행:  yolo/bin/python localization/sim_localizer.py            # 전 시나리오 + 판정
#        yolo/bin/python localization/sim_localizer.py --plot     # 궤적 PNG 저장
#
# 시뮬레이션 요소:
#  - 차동구동 로봇이 경기장을 한 바퀴 도는 경로 (스타트 존 → 위 → 골대 앞 → 복귀)
#  - 오도메트리 체계 오차 (바퀴 반지름 +2%, 축간거리 -3%) + 백색 노이즈 → 드리프트 재현
#  - 라이다 레이캐스트: 벽 4면 + 이음부 요철 + 골대 박스 + 움직이는 상대 로봇
#  - 스캔 1회전(0.1s) 동안의 로봇 이동에 의한 왜곡 (intra-scan distortion)
#  - 스캔 노이즈, 원거리 드롭아웃(벽 높이 여유 4-9cm → 기울면 먼 벽을 넘김)
#  - 상대 로봇에 밀림(오도메트리 미기록) → relocalize 복구 경로 검증

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from wall_localizer_core import (LocalizerConfig, WallLocalizer, laser_to_base,
                                 se2_compose, transform_points, wrap_angle)

ARENA_W, ARENA_H = 4.0, 4.0
GOAL_BOX = (0.0, 0.0, 0.40, 0.40)          # 왼쪽-아래 골대 (xmin, ymin, xmax, ymax)
START_POSE = (3.8, 0.2, math.pi / 2)       # 오른쪽-아래 스타트 존 중심, 위쪽(+y) 향함
N_BEAMS = 500                              # RPLidar C1: 5kHz / 10Hz ≈ 500점
SCAN_HZ = 10.0
DT = 0.02

WAYPOINTS = [(3.8, 3.2), (2.0, 3.4), (0.7, 3.2), (0.7, 1.0),
             (2.0, 0.7), (3.4, 1.2), (3.8, 0.8)]

JOINT_STEP = 1.33    # 벽 이음부 간격 [m] — 0714 실측 사진: 벽 4m가 약 1.33m 판재 3장
JOINT_HALF = 0.015   # 이음부 폭 절반 [m]
JOINT_DEPTH = 0.008  # 안쪽 돌출 [m]

# 0714 실측 사진 반영: 판재 이음부마다 알루미늄 평판 브라켓(~10cm 폭), 코너에는
# 앵글 브라켓(~8cm). 매끈한 금속 = 정반사 → 입사각이 법선에서 벗어나면 되반사가
# 거의 없어 드롭아웃되고, 가끔 거울 반사로 먼 벽을 보고 오는 멀티패스 고스트가 생긴다.
BRACKET_HALF = 0.05      # 이음부 브라켓 반폭 [m]
BRACKET_CORNER = 0.08    # 코너 앵글 브라켓 폭 [m]
BRACKET_SPEC_COS = math.cos(math.radians(12.0))  # 이 입사각 안쪽만 정상 되반사


def raycast(origin, heading, beam_angles, goal_box, opp_center, opp_r,
            upside_down=False, brackets=False, rng=None):
    """라이다 원점에서 전 빔 레이캐스트 → 거리 배열 (맞은 게 없으면 inf).

    upside_down: 뒤집어 장착하면 빔 각도가 거울상이 됨 (dirs = heading - angle).
    brackets:    금속 브라켓 정반사 아티팩트 (드롭아웃/멀티패스 고스트) 적용.
    """
    ox, oy = origin
    dirs = heading - beam_angles if upside_down else heading + beam_angles
    dx, dy = np.cos(dirs), np.sin(dirs)
    t = np.full(len(beam_angles), np.inf)
    bracket_bad = np.zeros(len(beam_angles), dtype=bool)  # 정반사로 되반사 없는 빔

    # 벽 4면 (안쪽에서 바깥으로 나가는 빔만 해당 벽에 닿음)
    for axis, coord, d, o in ((0, 0.0, dx, ox), (0, ARENA_W, dx, ox),
                              (1, 0.0, dy, oy), (1, ARENA_H, dy, oy)):
        with np.errstate(divide='ignore', invalid='ignore'):
            ti = (coord - o) / d
        # 다른 축 교차점이 경기장 범위 안이어야 함
        other = (oy + ti * dy) if axis == 0 else (ox + ti * dx)
        lim = ARENA_H if axis == 0 else ARENA_W
        valid = (ti > 1e-6) & np.isfinite(ti) & (other >= -0.01) & (other <= lim + 0.01)
        # 이음부: 벽면 위 위치가 이음부 구간이면 요철만큼 살짝 앞에서 맞음
        along = np.nan_to_num(other, nan=-1.0)
        near_joint = np.abs(along % JOINT_STEP) < JOINT_HALF
        near_joint |= np.abs(along % JOINT_STEP - JOINT_STEP) < JOINT_HALF
        ti_adj = np.where(valid & near_joint, np.maximum(ti - JOINT_DEPTH, 1e-6), ti)
        win = valid & (ti_adj < t)
        if brackets:
            # 이 벽의 브라켓 구간: 이음부 평판 + 코너 앵글
            on_bracket = near_joint.copy()
            for j in np.arange(JOINT_STEP, lim - 1e-6, JOINT_STEP):
                on_bracket |= np.abs(along - j) < BRACKET_HALF
            on_bracket |= (along >= -0.01) & (along < BRACKET_CORNER)
            on_bracket |= (along > lim - BRACKET_CORNER) & (along <= lim + 0.01)
            inc_cos = np.abs(d)  # 벽 법선(축방향)과 빔의 |cos|
            spec = valid & on_bracket & (inc_cos < BRACKET_SPEC_COS)
            bracket_bad = np.where(win, spec, bracket_bad)
        t = np.where(win, ti_adj, t)

    if brackets and rng is not None:
        # 정반사 빔: 70% 드롭아웃, 20% 멀티패스 고스트(1.3~2.2x 먼 거리), 10% 정상
        u = rng.random(len(t))
        ghost = bracket_bad & (u >= 0.7) & (u < 0.9)
        dropped = bracket_bad & (u < 0.7)
        t = np.where(ghost, t * rng.uniform(1.3, 2.2, len(t)), t)
        t = np.where(dropped, np.inf, t)

    # 골대 박스 (slab 법)
    if goal_box is not None:
        xmin, ymin, xmax, ymax = goal_box
        with np.errstate(divide='ignore', invalid='ignore'):
            tx1, tx2 = (xmin - ox) / dx, (xmax - ox) / dx
            ty1, ty2 = (ymin - oy) / dy, (ymax - oy) / dy
        tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
        tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
        hit = (tmax >= tmin) & (tmin > 1e-6)
        t = np.where(hit & (tmin < t), tmin, t)

    # 상대 로봇 (원)
    if opp_center is not None:
        cx, cy = opp_center
        fx, fy = ox - cx, oy - cy
        b = fx * dx + fy * dy
        cc = fx * fx + fy * fy - opp_r ** 2
        disc = b * b - cc
        ok = disc >= 0
        tc = -b - np.sqrt(np.maximum(disc, 0))
        hit = ok & (tc > 1e-6)
        t = np.where(hit & (tc < t), tc, t)

    return t


def diff_drive_step(pose, v, w, dt):
    x, y, th = pose
    x += v * math.cos(th + w * dt / 2) * dt
    y += v * math.sin(th + w * dt / 2) * dt
    return (x, y, wrap_angle(th + w * dt))


def waypoint_control(pose, wp):
    """단순 P 제어: 목표점 향해 회전 후 전진."""
    dx, dy = wp[0] - pose[0], wp[1] - pose[1]
    dist = math.hypot(dx, dy)
    ang_err = wrap_angle(math.atan2(dy, dx) - pose[2])
    w = max(-1.5, min(1.5, 2.5 * ang_err))
    v = 0.35 if abs(ang_err) < 0.5 else 0.0
    return v, w, dist


def in_grace(t, windows):
    return any(t0 <= t <= t1 for t0, t1 in windows)


def run_scenario(name, cfg_sim, seed=0, plot=False):
    rng = np.random.default_rng(seed)
    laser_pose = cfg_sim['laser_pose']          # base 기준 라이다 (x, y, yaw)
    upside_down = cfg_sim.get('upside_down', False)
    loc = WallLocalizer(LocalizerConfig())

    true_pose = START_POSE
    odom_pose = (0.0, 0.0, 0.0)                 # odom 원점 = 시작 자세
    # 오도메트리 체계 오차 (실제 로봇의 캘리브레이션 잔여 오차 재현)
    scale_v, scale_w = 1.02, 0.97

    init_err = cfg_sim.get('init_error', (0.0, 0.0, 0.0))
    guess = (START_POSE[0] + init_err[0], START_POSE[1] + init_err[1],
             START_POSE[2] + init_err[2])
    loc.set_pose(*guess, odom_pose=odom_pose)

    beam_angles = np.linspace(-math.pi, math.pi, N_BEAMS, endpoint=False)
    steps_per_scan = int(round(1.0 / SCAN_HZ / DT))

    wp_i, step, t_sim = 0, 0, 0.0
    records = []                                # (t, 위치오차, yaw오차)
    odom_errs, times = [], []
    rejects, updates, reloc_ok, cov_bad = 0, 0, 0, 0
    traj_true, traj_est = [], []
    blackout = cfg_sim.get('blackout', None)    # (시작s, 끝s) 스캔 두절 구간
    push = cfg_sim.get('push', None)            # (시각s, dx, dy) 밀림 (오도메트리 미기록)
    pushed = False
    grace = cfg_sim.get('grace', [])            # 평가 제외 구간 (재수렴 유예)
    pose_hist = [true_pose] * steps_per_scan    # 스캔 1회전 동안의 자세 (왜곡 재현용)

    while wp_i < len(WAYPOINTS) and step < 6000:
        v, w, dist = waypoint_control(true_pose, WAYPOINTS[wp_i])
        if dist < 0.12:
            wp_i += 1
            continue
        true_pose = diff_drive_step(true_pose, v, w, DT)
        if push and not pushed and t_sim >= push[0]:
            true_pose = (true_pose[0] + push[1], true_pose[1] + push[2],
                         true_pose[2])
            pushed = True
        v_m = v * scale_v + rng.normal(0, 0.01)
        w_m = w * scale_w + rng.normal(0, 0.02)
        odom_pose = diff_drive_step(odom_pose, v_m, w_m, DT)
        pose_hist.append(true_pose)
        if len(pose_hist) > steps_per_scan:
            pose_hist.pop(0)
        step += 1
        t_sim += DT

        if step % steps_per_scan != 0:
            continue

        # --- 스캔 생성: 1회전(0.1s) 동안 로봇이 움직이므로 빔을 시간 구간별로 나눠
        #     각 구간의 실제 자세에서 레이캐스트 (intra-scan 왜곡 재현) ---
        opp = None
        if cfg_sim.get('opponent', False):
            opp = (2.0 + 1.2 * math.sin(0.3 * t_sim), 2.0 + 1.2 * math.cos(0.23 * t_sim))
        goal = GOAL_BOX if cfg_sim.get('goal_box', False) else None
        ranges = np.empty(N_BEAMS)
        chunk = N_BEAMS // steps_per_scan
        for ci in range(steps_per_scan):
            lx, ly, lyaw = se2_compose(pose_hist[ci], laser_pose)
            s0, s1 = ci * chunk, (ci + 1) * chunk if ci < steps_per_scan - 1 else N_BEAMS
            ranges[s0:s1] = raycast((lx, ly), lyaw, beam_angles[s0:s1],
                                    goal, opp, 0.15, upside_down,
                                    brackets=cfg_sim.get('brackets', False), rng=rng)

        ranges = ranges + rng.normal(0, cfg_sim.get('noise', 0.008), N_BEAMS)
        far_p = cfg_sim.get('far_dropout_p', 0.25)
        far_d = cfg_sim.get('far_dropout_dist', 3.0)
        drop = (ranges > far_d) & (rng.random(N_BEAMS) < far_p)
        drop |= rng.random(N_BEAMS) < cfg_sim.get('rand_dropout_p', 0.02)
        if blackout and blackout[0] <= t_sim <= blackout[1]:
            drop[:] = True
        ranges = np.where(drop, np.inf, ranges)

        # --- 노드가 하는 전처리 재현: 거리 필터 + laser→base 변환 ---
        valid = np.isfinite(ranges) & (ranges > 0.15) & (ranges < 6.0)
        pts_base = laser_to_base(ranges[valid], beam_angles[valid], laser_pose,
                                 upside_down)

        t0 = time.perf_counter()
        res = loc.update(pts_base, odom_pose)
        if not res.accepted and loc.consecutive_rejects >= 5 and len(pts_base) > 100:
            if loc.relocalize(pts_base, odom_pose):
                reloc_ok += 1
        times.append(time.perf_counter() - t0)

        updates += 1
        rejects += 0 if res.accepted else 1
        if res.accepted:
            diag = np.diag(res.covariance)
            if not (np.all(diag > 0) and math.sqrt(diag[0]) < 0.05
                    and math.sqrt(diag[1]) < 0.05):
                cov_bad += 1
        est = loc.pose
        records.append((t_sim,
                        math.hypot(est[0] - true_pose[0], est[1] - true_pose[1]),
                        abs(wrap_angle(est[2] - true_pose[2]))))
        odom_only = se2_compose(START_POSE, odom_pose)
        odom_errs.append(math.hypot(odom_only[0] - true_pose[0],
                                    odom_only[1] - true_pose[1]))
        traj_true.append(true_pose[:2])
        traj_est.append(est[:2])

    ts = np.array([r[0] for r in records])
    errs = np.array([r[1] for r in records])
    yaw_errs = np.array([r[2] for r in records])
    eval_mask = np.array([not in_grace(t, grace) for t in ts])
    e, ye = errs[eval_mask], yaw_errs[eval_mask]
    grace_max = float(errs[~eval_mask].max()) if (~eval_mask).any() else 0.0

    stats = dict(name=name,
                 mean=e.mean(), p95=np.percentile(e, 95), max=e.max(),
                 yaw_mean=math.degrees(ye.mean()), yaw_max=math.degrees(ye.max()),
                 grace_max=grace_max, odom_final=odom_errs[-1],
                 rejects=rejects, updates=updates, reloc_ok=reloc_ok,
                 cov_bad=cov_bad, ms=1000 * float(np.mean(times)))

    if plot:
        plot_traj(name, traj_true, traj_est)
    return stats


def plot_traj(name, traj_true, traj_est):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tt, te = np.array(traj_true), np.array(traj_est)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(tt[:, 0], tt[:, 1], 'g-', label='true')
    ax.plot(te[:, 0], te[:, 1], 'r--', label='estimate')
    ax.add_patch(plt.Rectangle((0, 0), ARENA_W, ARENA_H, fill=False))
    ax.add_patch(plt.Rectangle(GOAL_BOX[:2], GOAL_BOX[2], GOAL_BOX[3],
                               fill=True, alpha=0.3, color='gray'))
    ax.add_patch(plt.Rectangle((3.6, 0.0), 0.4, 0.4, fill=True, alpha=0.3, color='blue'))
    ax.set_xlim(-0.3, 4.3); ax.set_ylim(-0.3, 4.3)
    ax.set_aspect('equal'); ax.legend(); ax.set_title(name)
    out = os.path.join(os.environ.get('SIM_PLOT_DIR', '/tmp'), f'sim_{name}.png')
    fig.savefig(out, dpi=100)
    print(f'  플롯 저장: {out}')


SCENARIOS = {
    # A: 기본 — 벽 + 이음부만
    'A_nominal': dict(laser_pose=(0.10, 0.0, 0.0)),
    # B: 골대 + 상대 로봇 등장
    'B_goal_opponent': dict(laser_pose=(0.10, 0.0, 0.0), goal_box=True, opponent=True),
    # C: 초기 배치 오차 (5cm, -5cm, +5°)
    'C_init_offset': dict(laser_pose=(0.10, 0.0, 0.0), goal_box=True, opponent=True,
                          init_error=(0.05, -0.05, math.radians(5))),
    # D: 악조건 — 노이즈 크고 원거리 드롭아웃 심함 (라이다 기울어짐 재현)
    'D_harsh': dict(laser_pose=(0.10, 0.0, 0.0), goal_box=True, opponent=True,
                    noise=0.015, far_dropout_p=0.5, far_dropout_dist=2.5,
                    rand_dropout_p=0.05),
    # E: 라이다를 뒤쪽 보게 장착 (yaw 180° 파라미터 검증)
    'E_mount_180': dict(laser_pose=(-0.05, 0.02, math.pi), goal_box=True, opponent=True),
    # F: 1.5초 스캔 두절 → 오도메트리 유지 후 복귀 (두절+1s는 유예)
    'F_blackout': dict(laser_pose=(0.10, 0.0, 0.0), goal_box=True, opponent=True,
                       blackout=(15.0, 16.5), grace=[(15.0, 17.5)]),
    # G: 상대에게 밀림 30cm (오도메트리 미기록) → 거부 누적 → relocalize 복구
    'G_push': dict(laser_pose=(0.10, 0.0, 0.0), goal_box=True, opponent=True,
                   push=(12.0, 0.22, -0.20), grace=[(12.0, 14.0)]),
    # H: 원거리 섹터 통째 손실 — 2.8m 이상 벽 반사가 전혀 안 옴 (심한 기울어짐)
    'H_sector_loss': dict(laser_pose=(0.10, 0.0, 0.0), goal_box=True, opponent=True,
                          far_dropout_p=1.0, far_dropout_dist=2.8),
    # I: 라이다 뒤집어 장착 + yaw 30° (upside_down 경로 검증)
    'I_upside_down': dict(laser_pose=(0.10, 0.0, math.radians(30)), goal_box=True,
                          opponent=True, upside_down=True),
    # J: 금속 브라켓 정반사 (0714 실측: 이음부 평판 + 코너 앵글) — 드롭아웃/고스트
    'J_brackets': dict(laser_pose=(0.10, 0.0, 0.0), goal_box=True, opponent=True,
                       brackets=True),
    # K: 브라켓 + 악조건 동시 (노이즈/원거리 드롭아웃까지 겹친 최악 케이스)
    'K_brackets_harsh': dict(laser_pose=(0.10, 0.0, 0.0), goal_box=True, opponent=True,
                             brackets=True, noise=0.015, far_dropout_p=0.5,
                             far_dropout_dist=2.5, rand_dropout_p=0.05),
    # L: 실측 장착 자세 (localization/params.yaml 2026-07-14 실측: 전방 1cm,
    #    왼쪽 바퀴 중심에서 오른쪽 17cm = 중심 기준 +0.5cm) — 배포 구성 그대로 검증
    'L_measured_mount': dict(laser_pose=(0.01, 0.005, 0.0), goal_box=True, opponent=True,
                             brackets=True),
}

LIMITS = dict(mean=0.03, p95=0.06, max=0.12, yaw_mean=2.0, ms=15.0, grace_max=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--scenario', default='all')
    args = ap.parse_args()

    names = list(SCENARIOS) if args.scenario == 'all' else [args.scenario]
    print(f'{"시나리오":<16} {"평균":>7} {"p95":>7} {"최대":>7} {"yaw평균":>7} '
          f'{"yaw최대":>7} {"유예최대":>8} {"odom최종":>8} {"거부":>7} {"ms":>5}  판정')
    all_ok = True
    for name in names:
        st = run_scenario(name, SCENARIOS[name], plot=args.plot)
        ok = (st['mean'] <= LIMITS['mean'] and st['p95'] <= LIMITS['p95']
              and st['max'] <= LIMITS['max'] and st['yaw_mean'] <= LIMITS['yaw_mean']
              and st['ms'] <= LIMITS['ms'] and st['grace_max'] <= LIMITS['grace_max']
              and st['cov_bad'] == 0)
        if name == 'G_push':
            ok = ok and st['reloc_ok'] >= 1   # 복구 경로가 실제로 실행됐는지
        all_ok &= ok
        print(f'{st["name"]:<16} {st["mean"]*100:6.1f}cm {st["p95"]*100:5.1f}cm '
              f'{st["max"]*100:5.1f}cm {st["yaw_mean"]:6.2f}° {st["yaw_max"]:6.2f}° '
              f'{st["grace_max"]*100:6.1f}cm {st["odom_final"]*100:6.1f}cm '
              f'{st["rejects"]:3d}/{st["updates"]:<3d} {st["ms"]:4.1f}  '
              f'{"PASS" if ok else "FAIL"}')

    print(f'\n기준: 평균≤{LIMITS["mean"]*100:.0f}cm p95≤{LIMITS["p95"]*100:.0f}cm '
          f'최대≤{LIMITS["max"]*100:.0f}cm yaw평균≤{LIMITS["yaw_mean"]}° '
          f'업데이트≤{LIMITS["ms"]:.0f}ms 공분산이상=0 '
          f'(유예 구간은 재수렴 중 최대 {LIMITS["grace_max"]*100:.0f}cm 허용, '
          f'G는 relocalize 성공 필수)')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
