# SNU-mech-AI-robotics-challenge
서울대학교 학부생 AI 로봇 챌린지

---

# Polyhedra detection (cube · octahedron · dodecahedron · icosahedron)

A lightweight YOLO11n detector for 4 types of 3D-printed solids, trained entirely on
synthetic images generated in **Isaac Sim Replicator**, and exported to **ONNX +
TensorRT** for the **Jetson Orin Nano**.

| STL (`datasets/`) | class id | name | faces |
|---|---|---|---|
| `6C1.STL`        | 0 | cube         | 6  |
| `8C1.STL`        | 1 | octahedron   | 8  |
| `12C1_Fixed.STL` | 2 | dodecahedron | 12 |
| `20C1.STL`       | 3 | icosahedron  | 20 |

## Layout

```
configs/
  classes.py            # single source of truth for the class<->id mapping
  polyhedra.yaml        # YOLO dataset config (4 classes)
isaac/                  # runs in Isaac Sim's python (NOT the yolo venv)
  convert_stl_to_usd.py # STL -> USD, mm->m, re-centered   (run once)
  generate_replicator.py# domain-randomized scenes -> YOLO dataset
  assets/usd/           # produced USD meshes
scripts/                # runs in the local yolo/ venv
  train.py              # YOLO11n training
  detect.py             # inference (.pt / .onnx / .engine, image / video / camera)
  export_jetson.py      # ONNX + TensorRT export
datasets/polyhedra/     # generated images/ + labels/  (train/ val/)
```

## Pipeline

### 1. Generate synthetic data — on a machine with Isaac Sim

Isaac Sim is **not** installed in this repo's venv; install it via the Omniverse
launcher or `pip install isaacsim` (4.x). Then:

```bash
# from a machine with Isaac Sim, in this repo root
python isaac/convert_stl_to_usd.py                       # STL -> isaac/assets/usd/*.usd
python isaac/generate_replicator.py --frames 4000 --val-ratio 0.15
```

This randomizes pose (full SO(3)), scale, object count, camera orbit, lighting,
ground colour and per-object PBR colour, and writes Ultralytics YOLO labels directly
(`class cx cy w h`, normalized). Tight 2D boxes come from the
`bounding_box_2d_tight` annotator; off-frame and <25%-visible objects are dropped.
Start at ~4k frames; scale to 10k+ if val mAP plateaus.

> Isaac Sim's Replicator API drifts between versions. If `rep.randomizer.color` or a
> light attribute name errors on your build, that line is the thing to adjust — the
> writer and scene structure are version-stable.

### 2. Train — local `yolo/` venv (RTX 4070 Ti Super)

```bash
yolo/bin/python scripts/train.py --epochs 100 --batch 32
# -> runs/detect/polyhedra/weights/best.pt
```

### 3. Inference

```bash
yolo/bin/python scripts/detect.py --weights runs/detect/polyhedra/weights/best.pt \
    --source path/to/photo.png --save
```

### 4. Export for Jetson Orin Nano

```bash
# dev box: portable ONNX
yolo/bin/python scripts/export_jetson.py --weights runs/detect/polyhedra/weights/best.pt --onnx

# ON the Orin Nano (builds a hardware-specific FP16 engine):
python scripts/export_jetson.py --weights best.pt --engine --half
python scripts/detect.py --weights best.engine --source 0     # live camera
```

The TensorRT `.engine` is tied to the Jetson's GPU/TensorRT version, so build it on
the device. FP16 (`--half`) is the recommended speed/accuracy trade-off on Orin Nano.

## Sim-to-real notes

- Training uses strong HSV/brightness jitter (`train.py`) so the model keys on the
  solids' **geometry**, not render-specific colour — important since data is synthetic.
- If real-world accuracy lags, add: more background/texture variety in Replicator,
  realistic clutter/occluders, motion blur, and a small set of **real** labelled
  photos for fine-tuning.
- Validate on real photos before trusting deployment metrics; synthetic-only val mAP
  is optimistic.
```

## 팀 셋업 가이드 (Ubuntu / WSL 공통)

이 repo에는 **코드·설정·학습된 모델 가중치(`models/`)·README·Claude 채팅 기록**이 들어 있어
clone 후 바로 추론을 돌려볼 수 있습니다.
대용량 데이터는 git에서 제외되어 있으므로(`.gitignore` 참고) 별도 채널(구글 드라이브/USB 등)로 받아야 합니다:

| 폴더 | 내용 | 용량 |
|---|---|---|
| `datasets/` | 학습 데이터셋 | ~34G |
| `yolo/` | 파이썬 venv (각자 새로 만들 것) | — |
| `runs/` | 학습 결과 | ~143M |
| `capture/` | 실촬영 영상/프레임 | ~830M |

TensorRT `.engine`은 기기(GPU/TensorRT 버전) 종속이라 커밋하지 않습니다 — Jetson에서 직접 빌드하세요.

### 새 컴퓨터에서 시작하기

```bash
git clone <팀-레포-URL> joon
cd joon
bash scripts/localize_paths.sh   # configs의 데이터셋 절대경로를 내 경로로 자동 치환
# 학습/데이터 작업이 필요하면 datasets/ 등을 별도 채널로 받아 같은 위치에 배치
```

### GitHub 인증 (push하려면 토큰 필요)

`git push` 시 비밀번호를 물어보는데, **GitHub 계정 비밀번호가 아니라 Personal Access Token(PAT)을 입력해야 합니다** (GitHub은 2021년부터 비밀번호 인증을 막았습니다).

1. github.com 로그인 → 우상단 프로필 → **Settings** → 왼쪽 맨 아래 **Developer settings**
   → **Personal access tokens** → **Tokens (classic)** → **Generate new token (classic)**
2. Note에 아무 이름, Expiration은 적당히(예: 90일), 권한은 **`repo`만 체크** → Generate
3. 생성된 `ghp_...` 문자열을 복사해두기 (창을 닫으면 다시 못 봄)
4. push할 때:
   ```
   Username: <내 GitHub 아이디>
   Password: <복사한 ghp_... 토큰 붙여넣기>
   ```

매번 입력하기 번거로우면 `git config credential.helper store`를 한 번 실행해두면
다음 입력부터 저장됩니다 (단, 토큰이 `~/.git-credentials`에 평문 저장되므로 공용 컴퓨터에서는 금지).
또한 push하려면 본인 계정이 이 레포의 **Collaborator**로 초대·수락되어 있어야 합니다 (403 에러가 나면 이것부터 확인).

### 기타 참고

- venv(`yolo/`)는 커밋되지 않으므로 각자 생성: `python3 -m venv yolo && yolo/bin/pip install ultralytics`
- WSL 사용자 주의: 프로젝트를 반드시 **리눅스 파일시스템**(`~/joon`)에 두세요.
  `/mnt/c/...`(윈도우 드라이브)에 두면 학습 I/O가 매우 느리고 권한 문제가 생깁니다.
- 줄바꿈은 `.gitattributes`로 LF로 강제되어 있어 WSL/Ubuntu 간 diff 오염이 없습니다.
- Claude Code 채팅 기록 복원 방법은 [`_claude_history/README.md`](_claude_history/README.md) 참고.
