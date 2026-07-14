#!/usr/bin/env python3
# 벽 기반 2D 위치추정 코어 (ROS 의존성 없음 — WSL에서 시뮬레이션 검증 가능)
#
# 전제: 4m x 4m 정사각형 경기장, 벽 4면이 라이다 스캔 평면(~20-25cm)에 항상 보임.
# 원리: 오도메트리로 예측 → 스캔 점들을 맵 좌표로 옮겨 가장 가까운 벽 직선에 대응시키고
#       점-직선 거리(Huber 가중)를 Gauss-Newton으로 최소화해 (x, y, yaw) 보정.
# 벽 이음부 요철·골대·상대 로봇 점들은 대응 임계값/Huber로 아웃라이어 처리된다.
#
# 좌표계 (위에서 본 기준):
#   map 원점 = 경기장 안쪽 왼쪽-아래 모서리, x → 오른쪽, y → 위쪽, yaw는 +x에서 반시계.
#   스타트 존 = 오른쪽-아래 40x40cm → 시작 자세 (3.8, 0.2, +90°) 근방.

import math
from dataclasses import dataclass, field

import numpy as np


def wrap_angle(a):
    """각도를 (-pi, pi]로 정규화."""
    return math.atan2(math.sin(a), math.cos(a))


def se2_compose(a, b):
    """SE(2) 합성 a ⊕ b. 각각 (x, y, th)."""
    ax, ay, ath = a
    bx, by, bth = b
    c, s = math.cos(ath), math.sin(ath)
    return (ax + c * bx - s * by,
            ay + s * bx + c * by,
            wrap_angle(ath + bth))


def se2_inverse(a):
    """SE(2) 역원."""
    ax, ay, ath = a
    c, s = math.cos(ath), math.sin(ath)
    return (-(c * ax + s * ay),
            -(-s * ax + c * ay),
            wrap_angle(-ath))


def transform_points(points, pose):
    """(N,2) 점들을 pose=(x,y,th)로 변환."""
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]])
    return points @ R.T + np.array([x, y])


def laser_to_base(ranges, angles, laser_pose, upside_down=False):
    """라이다 극좌표 → base_link 평면 좌표.

    laser_pose: base_link 기준 라이다 장착 (x, y, yaw).
    upside_down: 라이다를 뒤집어 장착했으면 True (스캔 y축 반전).
    """
    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)
    if upside_down:
        ys = -ys
    pts = np.stack([xs, ys], axis=1)
    return transform_points(pts, laser_pose)


@dataclass
class LocalizerConfig:
    arena_w: float = 4.0            # 경기장 안쪽 가로 [m]
    arena_h: float = 4.0            # 경기장 안쪽 세로 [m]
    assoc_thresh: float = 0.15      # 벽 대응 허용 거리 [m] — 이보다 멀면 아웃라이어
    huber_delta: float = 0.05       # Huber 손실 전환점 [m]
    iterations: int = 5             # GN 반복 횟수 (매 반복 재대응)
    max_correction_xy: float = 0.30     # 예측 대비 위치 보정 한계 [m]
    max_correction_yaw: float = math.radians(12.0)  # 예측 대비 각도 보정 한계
    min_inlier_ratio: float = 0.40  # 유효 점 대비 인라이어 최소 비율
    min_points: int = 60            # 유효 스캔 점 최소 개수
    min_axis_inliers: int = 20      # x벽/y벽 각각 최소 인라이어 (관측 가능성 확인)
    max_rms: float = 0.08           # 인라이어 잔차 RMS 상한 [m]
    sigma_scan: float = 0.01        # 공분산 스케일용 스캔 노이즈 [m]


@dataclass
class UpdateResult:
    pose: tuple                     # 보정된 (x, y, th) — 거부 시 예측값
    accepted: bool                  # 스캔 보정 채택 여부
    inlier_ratio: float
    rms: float                      # 인라이어 잔차 RMS [m]
    n_valid: int                    # 유효 스캔 점 수
    n_inliers: int
    correction: tuple = (0.0, 0.0, 0.0)   # 예측 → 보정 변화량
    covariance: np.ndarray = field(default_factory=lambda: np.eye(3))
    reject_reason: str = ''


class WallLocalizer:
    """오도메트리 예측 + 벽 4면 점-직선 정합 보정."""

    def __init__(self, config: LocalizerConfig = None):
        self.cfg = config or LocalizerConfig()
        self._T_map_odom = None     # map ← odom 보정 변환 (x, y, th)
        self._pose = None           # 마지막 base_link 자세 (map 기준)
        self.consecutive_rejects = 0

    @property
    def initialized(self):
        return self._T_map_odom is not None

    @property
    def T_map_odom(self):
        return self._T_map_odom

    @property
    def pose(self):
        return self._pose

    def set_pose(self, x, y, th, odom_pose):
        """초기/재설정: map 기준 자세와 그 시점의 odom 자세를 함께 준다."""
        self._pose = (x, y, wrap_angle(th))
        self._T_map_odom = se2_compose(self._pose, se2_inverse(odom_pose))
        self.consecutive_rejects = 0

    def predict(self, odom_pose):
        """odom 자세로 map 기준 예측 자세 계산."""
        return se2_compose(self._T_map_odom, odom_pose)

    def update(self, points_base, odom_pose):
        """스캔 1회분으로 자세 갱신.

        points_base: (N,2) base_link 기준 스캔 점 (유효 거리 필터링 완료 상태)
        odom_pose:   스캔 시점의 odom 기준 base_link 자세 (x, y, th)
        """
        if not self.initialized:
            raise RuntimeError('set_pose()로 초기 자세를 먼저 설정하세요')

        cfg = self.cfg
        pred = self.predict(odom_pose)
        n_valid = len(points_base)

        if n_valid < cfg.min_points:
            return self._reject(pred, odom_pose, n_valid, 'too_few_points')

        pose, H, r, inlier, wall_idx = self._gauss_newton(points_base, pred)

        n_in = int(inlier.sum())
        inlier_ratio = n_in / n_valid
        rms = float(np.sqrt(np.mean(r[inlier] ** 2))) if n_in > 0 else float('inf')
        # x벽(0,1) / y벽(2,3) 각각 인라이어가 있어야 x/y 모두 관측 가능
        n_x = int(np.sum(inlier & (wall_idx < 2)))
        n_y = int(np.sum(inlier & (wall_idx >= 2)))
        corr = (pose[0] - pred[0], pose[1] - pred[1],
                wrap_angle(pose[2] - pred[2]))

        reason = ''
        if inlier_ratio < cfg.min_inlier_ratio:
            reason = 'low_inlier_ratio'
        elif n_x < cfg.min_axis_inliers or n_y < cfg.min_axis_inliers:
            reason = 'axis_unobservable'
        elif rms > cfg.max_rms:
            reason = 'high_rms'
        elif math.hypot(corr[0], corr[1]) > cfg.max_correction_xy:
            reason = 'large_xy_correction'
        elif abs(corr[2]) > cfg.max_correction_yaw:
            reason = 'large_yaw_correction'

        if reason:
            res = self._reject(pred, odom_pose, n_valid, reason)
            res.inlier_ratio = inlier_ratio
            res.rms = rms
            res.n_inliers = n_in
            return res

        # 채택: map←odom 보정 갱신
        self._pose = pose
        self._T_map_odom = se2_compose(pose, se2_inverse(odom_pose))
        self.consecutive_rejects = 0

        try:
            cov = cfg.sigma_scan ** 2 * np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = np.eye(3)
        return UpdateResult(pose=pose, accepted=True, inlier_ratio=inlier_ratio,
                            rms=rms, n_valid=n_valid, n_inliers=n_in,
                            correction=corr, covariance=cov)

    def relocalize(self, points_base, odom_pose, search_xy=0.35, search_yaw=math.radians(18),
                   step_xy=0.07, step_yaw=math.radians(3)):
        """예측 자세 주변 좌표 격자 탐색으로 재수렴 시도 (연속 거부 시 호출).

        정사각형 경기장은 90° 대칭이므로 탐색 범위를 ±18°로 제한해
        엉뚱한 대칭 자세로 점프하지 않게 한다. 후보는 예측에 가까운 순서로
        평가하므로 동점이면 예측에 가장 가까운 자세가 이긴다 — 관측 안 되는
        축이 극단 오프셋으로 끌려가는 것을 막는다. 채택 전에 update()와 같은
        게이트(축별 인라이어/RMS/보정 한계)를 모두 통과해야 하며, 실패하면
        오도메트리 예측을 유지한다 (탐색 범위 밖으로 밀린 경우 억지로 맞추지 않음).
        """
        cfg = self.cfg
        pred = self.predict(odom_pose)
        n = len(points_base)
        if n < cfg.min_points:
            return False

        vals_xy = np.arange(-search_xy, search_xy + 1e-9, step_xy)
        vals_th = np.arange(-search_yaw, search_yaw + 1e-9, step_yaw)
        cands = [(abs(dx) + abs(dy) + 2.0 * abs(dth), dx, dy, dth)
                 for dth in vals_th for dx in vals_xy for dy in vals_xy]
        cands.sort(key=lambda c: c[0])

        best, best_score = None, -1
        for _, dx, dy, dth in cands:
            cand = (pred[0] + dx, pred[1] + dy, wrap_angle(pred[2] + dth))
            pm = transform_points(points_base, cand)
            d = self._wall_distances(pm)
            score = int(np.sum(np.abs(d).min(axis=1) < 0.05))
            if score > best_score:
                best, best_score = cand, score
        if best is None:
            return False

        # 최적 후보에서 GN 마무리 후 update()와 동일한 게이트 적용
        pose, _, r, inlier, wall_idx = self._gauss_newton(points_base, best)
        n_in = int(inlier.sum())
        if n_in == 0:
            return False
        rms = float(np.sqrt(np.mean(r[inlier] ** 2)))
        n_x = int(np.sum(inlier & (wall_idx < 2)))
        n_y = int(np.sum(inlier & (wall_idx >= 2)))
        corr_xy = math.hypot(pose[0] - pred[0], pose[1] - pred[1])
        corr_th = abs(wrap_angle(pose[2] - pred[2]))
        if (n_in / n < cfg.min_inlier_ratio
                or n_x < cfg.min_axis_inliers or n_y < cfg.min_axis_inliers
                or rms > cfg.max_rms
                or corr_xy > search_xy + cfg.assoc_thresh
                or corr_th > search_yaw + math.radians(5)):
            return False

        self._pose = pose
        self._T_map_odom = se2_compose(pose, se2_inverse(odom_pose))
        self.consecutive_rejects = 0
        return True

    # --- 내부 ---

    def _wall_distances(self, pm):
        """각 점의 4개 벽 직선까지 부호 있는 거리 (N,4). 벽: x=0, x=W, y=0, y=H."""
        cfg = self.cfg
        return np.stack([pm[:, 0],
                         pm[:, 0] - cfg.arena_w,
                         pm[:, 1],
                         pm[:, 1] - cfg.arena_h], axis=1)

    def _gauss_newton(self, points, init_pose):
        cfg = self.cfg
        pose = init_pose
        n = len(points)
        r = np.zeros(n)
        inlier = np.zeros(n, dtype=bool)
        wall_idx = np.zeros(n, dtype=int)
        H = np.eye(3)

        for _ in range(cfg.iterations):
            x, y, th = pose
            c, s = math.cos(th), math.sin(th)
            px, py = points[:, 0], points[:, 1]
            pmx = c * px - s * py + x
            pmy = s * px + c * py + y

            D = np.stack([pmx, pmx - cfg.arena_w, pmy, pmy - cfg.arena_h], axis=1)
            absD = np.abs(D)
            wall_idx = absD.argmin(axis=1)
            r = D[np.arange(n), wall_idx]
            inlier = np.abs(r) < cfg.assoc_thresh
            if inlier.sum() < 3:
                break

            # Jacobian: 잔차 = n·p_map - c_wall,  n = (1,0) or (0,1)
            dpx_dth = -s * px - c * py
            dpy_dth = c * px - s * py
            is_x_wall = wall_idx < 2
            J = np.zeros((n, 3))
            J[is_x_wall, 0] = 1.0
            J[is_x_wall, 2] = dpx_dth[is_x_wall]
            J[~is_x_wall, 1] = 1.0
            J[~is_x_wall, 2] = dpy_dth[~is_x_wall]

            absr = np.abs(r)
            w = np.where(absr <= cfg.huber_delta, 1.0,
                         cfg.huber_delta / np.maximum(absr, 1e-12))
            w = w * inlier

            Jw = J * w[:, None]
            H = J.T @ Jw + 1e-9 * np.eye(3)
            g = Jw.T @ r
            try:
                delta = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                break
            pose = (pose[0] + delta[0], pose[1] + delta[1],
                    wrap_angle(pose[2] + delta[2]))
            if np.abs(delta).max() < 1e-5:
                break

        return pose, H, r, inlier, wall_idx

    def _reject(self, pred, odom_pose, n_valid, reason):
        # 거부 시에도 오도메트리 예측은 따라간다 (T_map_odom은 유지)
        self._pose = pred
        self.consecutive_rejects += 1
        return UpdateResult(pose=pred, accepted=False, inlier_ratio=0.0,
                            rms=float('inf'), n_valid=n_valid, n_inliers=0,
                            reject_reason=reason)
