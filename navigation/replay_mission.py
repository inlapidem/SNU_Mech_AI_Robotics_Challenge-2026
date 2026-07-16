#!/usr/bin/env python3
"""경기 시뮬레이션 리플레이 — 로봇이 실제로 어떻게 움직이는지 눈으로 본다.

sim_mission.run_match 는 매 프레임 hooks["tick"](t, world, mission) 을 부른다.
그 훅으로 시뮬레이터를 **한 줄도 고치지 않고** 로봇 포즈·물체 상태·미션 상태를
기록한 뒤, 위에서 내려다본 인터랙티브 HTML 리플레이(재생/일시정지/스크럽/속도,
로봇 궤적·카메라 시야, 물체/포획/하역, 실시간 상태·점수)로 렌더링한다.

실행:
  # seed 3, both 모드(세트1 형상 + 세트2 과일) 리플레이 HTML 생성 후 브라우저로 열기
  yolo/bin/python navigation/replay_mission.py --seed 3 --mode both \
      --out navigation/replays/match.html

  # 원시 궤적 JSON 만 (다른 도구로 렌더링)
  yolo/bin/python navigation/replay_mission.py --seed 3 --json > traj.json

  # 여러 시드 요약만 훑어 좋은 경기 고르기
  yolo/bin/python navigation/replay_mission.py --scan 0 12 --mode both
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_mission import FRUITS, run_match          # noqa: E402
from mission_fsm import (IDLE, TOUR, GOTO, APPROACH, CAPTURE, RETREAT,  # noqa: E402
                         TRANSPORT, DEPOSIT_SHED, DEPOSIT_PUSH,
                         DEPOSIT_RELEASE, PARK, DONE, DEPOSIT_REALIGN)
from nav_core import ArenaGeometry                 # noqa: E402

# 상태 순서 = HTML 뷰어의 색/라벨 인덱스와 1:1 대응 (뒤에만 추가할 것)
STATES = [IDLE, TOUR, GOTO, APPROACH, CAPTURE, RETREAT, TRANSPORT,
          DEPOSIT_SHED, DEPOSIT_PUSH, DEPOSIT_RELEASE, PARK, DONE,
          DEPOSIT_REALIGN]
STATE_IDX = {s: i for i, s in enumerate(STATES)}

SHAPE_CHOICES = ["octa", "dodeca", "icosa"]        # cube 는 공지 특수 모드


def _targets_for(mode, rng_seed):
    import random
    rng = random.Random(9000 + rng_seed)
    if mode == "set1":
        return {"set1": rng.choice(SHAPE_CHOICES)}
    if mode == "set1cube":
        return {"set1": "cube"}
    if mode == "set2":
        return {"set2": rng.choice(FRUITS)}
    if mode == "bothcube":
        return {"set1": "cube", "set2": rng.choice(FRUITS)}
    return {"set1": rng.choice(SHAPE_CHOICES), "set2": rng.choice(FRUITS)}


def record(seed, mode="both", cruise=0.20, duration=180.0, rec_every=2):
    """한 경기를 돌리며 프레임을 기록하고, 뷰어가 먹을 data dict 를 돌려준다."""
    geom = ArenaGeometry()
    targets = _targets_for(mode, seed)
    params = dict(cruise_v=cruise, eff_speed=cruise * 0.73)

    frames = []              # 로봇: [x_mm, y_mm, yaw_mrad, st, ir, pl, fr, tgt, t_ds]
    obj_meta = []            # 물체 정적 정보: {id,set,cls,target}
    obj_deltas = []          # 프레임별 변화분: [[idx, x_mm, y_mm, status], ...]
    obj0 = []               # 초기 물체 위치 [x_mm, y_mm]
    state = {"init": False}
    id_to_idx = {}
    prev = {}               # idx -> (x_mm, y_mm, status)
    counter = {"i": 0}

    def status_of(world, o):
        if o.get("stored") or o.get("done"):
            return 3        # 보관함 안착
        if o is world.payload:
            return 1        # 빈에 적재
        if o is world.front:
            return 2        # 입구에 문 물체(더블 캐리)
        return 0            # 경기장

    def tick(t, world, mission):
        i = counter["i"]
        counter["i"] += 1
        if i % rec_every != 0:
            return
        if not state["init"]:
            for idx, o in enumerate(world.objs):
                id_to_idx[o["id"]] = idx
                obj_meta.append(dict(id=o["id"], set=o["set"], cls=o["cls"],
                                     target=(o["cls"] == targets.get(o["set"]))))
                xi, yi = round(o["x"] * 1000), round(o["y"] * 1000)
                obj0.append([xi, yi])
                prev[idx] = (xi, yi, status_of(world, o))
            state["init"] = True

        x, y, yaw = world.pose
        tgt = getattr(mission, "_target", None)
        frames.append([
            round(x * 1000), round(y * 1000), round(yaw * 1000),
            STATE_IDX.get(mission.state, 0),
            1 if world.ir else 0,
            world.payload["id"] if world.payload else -1,
            world.front["id"] if world.front else -1,
            tgt["id"] if tgt else -1,
            round(t * 10),
        ])
        d = []
        for idx, o in enumerate(world.objs):
            cur = (round(o["x"] * 1000), round(o["y"] * 1000),
                   status_of(world, o))
            if cur != prev.get(idx):
                d.append([idx, cur[0], cur[1], cur[2]])
                prev[idx] = cur
        obj_deltas.append(d)

    res = run_match(seed=seed, targets=targets, duration=duration,
                    hooks={"tick": tick}, params_override=params)

    # 이벤트를 기록된 프레임 인덱스에 스냅
    fi_by_ds = {}
    for fi, fr in enumerate(frames):
        fi_by_ds.setdefault(fr[8], fi)
    events = []
    for (te, txt) in res["events"]:
        ds = round(te * 10)
        fi = fi_by_ds.get(ds)
        if fi is None:                       # 가장 가까운 프레임
            fi = min(range(len(frames)),
                     key=lambda k: abs(frames[k][8] - ds)) if frames else 0
        events.append({"fi": fi, "t": te, "txt": txt})

    def rect(r):
        return [r.x0, r.y0, r.x1, r.y1]

    data = {
        "meta": {
            "seed": seed, "mode": mode, "cruise": cruise,
            "duration": duration, "rec_every": rec_every, "dt": 0.05,
            "arena": [geom.arena_w, geom.arena_h],
            "start_zone": rect(geom.start_zone),
            "storage": rect(geom.storage),
            "sticker_zone": rect(geom.sticker_zone),
            "score_box": [0.04, 0.04, 0.36, 0.36],
            "targets": targets,
            "geom": {"body_r": 0.17, "pocket": 0.07, "front_off": 0.35,
                     "length": 0.34, "width": 0.31, "wheel_w": 0.38,
                     "rc_front": 0.165, "rc_back": 0.175,
                     "scoop_half": 0.073, "scoop_depth": 0.13,
                     "search_r": 2.6, "search_fov": 90.0,
                     "verify_r": 1.3, "verify_fov": 62.0},
            "result": {k: res[k] for k in
                       ("points", "good", "bad", "wall_hits", "spilled",
                        "front_slips", "end_state", "t_end", "deposit_times")},
        },
        "objects": obj_meta,
        "obj0": obj0,
        "frames": frames,
        "obj_deltas": obj_deltas,
        "events": events,
    }
    return data, res


# ----------------------------------------------------------------- HTML 렌더링

def render_html(data, fragment=False):
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    m = data["meta"]
    tsel = m["targets"]
    title = f"경기 리플레이 · seed {m['seed']} · {m['mode']}"
    core = _VIEWER_TEMPLATE.replace("/*__DATA__*/null", payload)
    if fragment:
        return core
    return (
        "<!doctype html>\n<html lang=\"ko\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        f"<title>{title}</title>\n</head>\n<body>\n{core}\n</body>\n</html>\n"
    )


_VIEWER_TEMPLATE = r"""
<style>
  :root{
    --bg:#0f141c; --panel:#151c27; --panel2:#1b2431; --line:#26313f;
    --grid:#1e2735; --ink:#e2e8f0; --muted:#8a97a8; --muted2:#697585;
    --accent:#ffab3d; --accent-ink:#ffce8a;
    --good:#3ecf8e; --bad:#ff5d63; --carry:#38c0cb;
    --c-cube:#9aa7b8; --c-octa:#a78bfa; --c-dodeca:#5b9bf0; --c-icosa:#818cf8;
    --c-apple:#ef4b4b; --c-orange:#f59e2c; --c-banana:#eacb2e; --c-pineapple:#d19a2e;
    --shadow:0 1px 0 rgba(255,255,255,.03), 0 8px 30px rgba(0,0,0,.35);
    --mono:ui-monospace,"JetBrains Mono","SF Mono",Menlo,Consolas,monospace;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
  }
  @media (prefers-color-scheme: light){
    :root{
      --bg:#eef1f5; --panel:#ffffff; --panel2:#f4f6f9; --line:#dde3ea;
      --grid:#e2e7ee; --ink:#182231; --muted:#5c6675; --muted2:#7a848f;
      --accent:#d97800; --accent-ink:#a85e00;
      --good:#0f9d63; --bad:#d63b41; --carry:#0e93a0;
      --c-cube:#6b7686; --c-octa:#7c5fe0; --c-dodeca:#2f74d0; --c-icosa:#5b62d8;
      --c-apple:#d63030; --c-orange:#d47b12; --c-banana:#b99310; --c-pineapple:#a9781a;
      --shadow:0 1px 2px rgba(20,30,50,.06), 0 8px 24px rgba(20,30,50,.10);
    }
  }
  :root[data-theme="dark"]{
    --bg:#0f141c; --panel:#151c27; --panel2:#1b2431; --line:#26313f;
    --grid:#1e2735; --ink:#e2e8f0; --muted:#8a97a8; --muted2:#697585;
    --accent:#ffab3d; --accent-ink:#ffce8a; --good:#3ecf8e; --bad:#ff5d63; --carry:#38c0cb;
    --c-cube:#9aa7b8; --c-octa:#a78bfa; --c-dodeca:#5b9bf0; --c-icosa:#818cf8;
    --c-apple:#ef4b4b; --c-orange:#f59e2c; --c-banana:#eacb2e; --c-pineapple:#d19a2e;
  }
  :root[data-theme="light"]{
    --bg:#eef1f5; --panel:#ffffff; --panel2:#f4f6f9; --line:#dde3ea;
    --grid:#e2e7ee; --ink:#182231; --muted:#5c6675; --muted2:#7a848f;
    --accent:#d97800; --accent-ink:#a85e00; --good:#0f9d63; --bad:#d63b41; --carry:#0e93a0;
    --c-cube:#6b7686; --c-octa:#7c5fe0; --c-dodeca:#2f74d0; --c-icosa:#5b62d8;
    --c-apple:#d63030; --c-orange:#d47b12; --c-banana:#b99310; --c-pineapple:#a9781a;
  }
  *{box-sizing:border-box}
  #rp{
    font-family:var(--sans); color:var(--ink); background:var(--bg);
    padding:18px; min-height:100%; line-height:1.45;
    -webkit-font-smoothing:antialiased;
  }
  #rp .wrap{max-width:1120px; margin:0 auto; display:flex; flex-direction:column; gap:14px}
  #rp h1{font-size:15px; font-weight:650; margin:0; letter-spacing:.01em}
  #rp .eyebrow{font-family:var(--mono); font-size:10.5px; letter-spacing:.16em;
    text-transform:uppercase; color:var(--muted); margin:0 0 3px}
  #rp .hdr{display:flex; justify-content:space-between; align-items:flex-end;
    gap:16px; flex-wrap:wrap}
  #rp .hdr .meta{display:flex; gap:16px; flex-wrap:wrap; font-family:var(--mono);
    font-size:11.5px; color:var(--muted)}
  #rp .hdr .meta b{color:var(--ink); font-weight:600}
  #rp .stage{display:grid; grid-template-columns:minmax(0,1fr) 288px; gap:14px}
  @media (max-width:820px){ #rp .stage{grid-template-columns:1fr} }
  #rp .card{background:var(--panel); border:1px solid var(--line);
    border-radius:12px; box-shadow:var(--shadow)}
  #rp .canvas-card{padding:12px; position:relative}
  #rp canvas{width:100%; height:auto; display:block; border-radius:7px; touch-action:none}
  #rp .side{display:flex; flex-direction:column; gap:14px; min-width:0}
  #rp .pad{padding:13px 14px}
  #rp .stat-grid{display:grid; grid-template-columns:1fr 1fr; gap:1px;
    background:var(--line); border-radius:10px; overflow:hidden; border:1px solid var(--line)}
  #rp .stat{background:var(--panel); padding:9px 11px}
  #rp .stat .k{font-family:var(--mono); font-size:9.5px; letter-spacing:.12em;
    text-transform:uppercase; color:var(--muted)}
  #rp .stat .v{font-family:var(--mono); font-size:20px; font-weight:600;
    font-variant-numeric:tabular-nums; margin-top:2px}
  #rp .stat .v small{font-size:11px; color:var(--muted); font-weight:500}
  #rp .v.good{color:var(--good)} #rp .v.bad{color:var(--bad)}
  #rp .state-row{display:flex; align-items:center; gap:9px; margin-bottom:11px}
  #rp .chip{font-family:var(--mono); font-size:11.5px; font-weight:600;
    padding:4px 10px; border-radius:999px; letter-spacing:.02em;
    background:color-mix(in srgb, var(--sc) 20%, transparent);
    color:var(--sc); border:1px solid color-mix(in srgb, var(--sc) 42%, transparent)}
  #rp .carrying{font-family:var(--mono); font-size:11px; color:var(--carry);
    display:flex; align-items:center; gap:6px}
  #rp .carrying .dot{width:7px;height:7px;border-radius:50%;background:var(--carry)}
  #rp .carrying.off{color:var(--muted2)} #rp .carrying.off .dot{background:var(--muted2)}
  #rp .sect-t{font-family:var(--mono); font-size:10px; letter-spacing:.14em;
    text-transform:uppercase; color:var(--muted); margin:0 0 9px}
  #rp .legend{display:flex; flex-direction:column; gap:7px}
  #rp .lg-row{display:flex; align-items:center; gap:8px; font-size:12px; color:var(--muted)}
  #rp .lg-row b{color:var(--ink); font-weight:550}
  #rp .swatch{width:13px;height:13px;flex:0 0 auto;display:inline-block}
  #rp .log{font-family:var(--mono); font-size:11px; height:132px; overflow-y:auto;
    display:flex; flex-direction:column; gap:3px; scrollbar-width:thin}
  #rp .log .ev{color:var(--muted2); display:flex; gap:8px; padding:2px 4px; border-radius:5px}
  #rp .log .ev .tt{color:var(--muted); flex:0 0 46px; text-align:right;
    font-variant-numeric:tabular-nums}
  #rp .log .ev.now{background:color-mix(in srgb,var(--accent) 16%,transparent); color:var(--ink)}
  #rp .log .ev.now .tt{color:var(--accent-ink)}
  /* transport */
  #rp .transport{display:flex; align-items:center; gap:13px; padding:12px 15px}
  #rp button.icon{width:38px;height:38px;flex:0 0 auto;border-radius:9px;
    border:1px solid var(--line); background:var(--panel2); color:var(--ink);
    cursor:pointer; display:grid; place-items:center; transition:.12s}
  #rp button.icon:hover{border-color:var(--accent); color:var(--accent)}
  #rp button.icon svg{width:16px;height:16px;fill:currentColor}
  #rp button.icon.play{background:var(--accent); color:#161007; border-color:var(--accent)}
  #rp button.icon.play:hover{filter:brightness(1.08); color:#161007}
  #rp .scrub-wrap{flex:1; display:flex; flex-direction:column; gap:5px; min-width:0}
  #rp .track{position:relative; height:22px; display:flex; align-items:center}
  #rp input[type=range].scrub{-webkit-appearance:none; appearance:none; width:100%;
    height:5px; border-radius:5px; background:var(--panel2); outline:none; cursor:pointer;
    border:1px solid var(--line)}
  #rp input[type=range].scrub::-webkit-slider-thumb{-webkit-appearance:none;
    width:15px;height:15px;border-radius:50%;background:var(--accent);
    border:2px solid var(--panel); cursor:pointer; box-shadow:0 0 0 1px var(--accent)}
  #rp input[type=range].scrub::-moz-range-thumb{width:15px;height:15px;border-radius:50%;
    background:var(--accent); border:2px solid var(--panel); cursor:pointer}
  #rp .ticks{position:absolute; left:0; right:0; top:1px; height:20px; pointer-events:none}
  #rp .tick{position:absolute; width:2px; height:9px; top:0; border-radius:2px; transform:translateX(-1px)}
  #rp .time{font-family:var(--mono); font-size:11.5px; color:var(--muted);
    font-variant-numeric:tabular-nums; display:flex; justify-content:space-between}
  #rp .time b{color:var(--ink); font-weight:600}
  #rp .rt{display:flex; align-items:center; gap:8px}
  #rp .speed{display:flex; gap:2px; background:var(--panel2); border:1px solid var(--line);
    border-radius:8px; padding:2px}
  #rp .speed button{font-family:var(--mono); font-size:11px; border:none; background:none;
    color:var(--muted); padding:4px 7px; border-radius:6px; cursor:pointer}
  #rp .speed button.on{background:var(--accent); color:#161007; font-weight:600}
  #rp .toggle{font-family:var(--mono); font-size:10.5px; color:var(--muted);
    display:flex; align-items:center; gap:6px; cursor:pointer; user-select:none;
    border:1px solid var(--line); border-radius:8px; padding:6px 9px; background:var(--panel2)}
  #rp .toggle.on{color:var(--accent); border-color:var(--accent)}
  #rp .toggle input{display:none}
  #rp .foot{font-family:var(--mono); font-size:10.5px; color:var(--muted2);
    text-align:center; padding-top:2px}
  @media (prefers-reduced-motion: reduce){ #rp *{scroll-behavior:auto} }
</style>

<div id="rp">
  <div class="wrap">
    <div class="hdr">
      <div>
        <p class="eyebrow">미션 시뮬레이션 리플레이</p>
        <h1 id="rp-title">경기 리플레이</h1>
      </div>
      <div class="meta" id="rp-meta"></div>
    </div>

    <div class="stage">
      <div class="card canvas-card">
        <canvas id="rp-cv"></canvas>
      </div>
      <div class="side">
        <div class="card pad">
          <div class="state-row">
            <span class="chip" id="rp-state">—</span>
            <span class="carrying off" id="rp-carry"><span class="dot"></span><span id="rp-carry-t">빈 비어있음</span></span>
          </div>
          <div class="stat-grid">
            <div class="stat"><div class="k">경과 / 총</div><div class="v"><span id="rp-t">0.0</span><small id="rp-tend"></small></div></div>
            <div class="stat"><div class="k">점수</div><div class="v" id="rp-pts">0</div></div>
            <div class="stat"><div class="k">정상 하역</div><div class="v good" id="rp-good">0</div></div>
            <div class="stat"><div class="k">오픽업 / 벽</div><div class="v bad" id="rp-bad">0</div></div>
          </div>
        </div>

        <div class="card pad">
          <p class="sect-t">범례</p>
          <div class="legend" id="rp-legend"></div>
        </div>

        <div class="card pad">
          <p class="sect-t">이벤트 로그</p>
          <div class="log" id="rp-log"></div>
        </div>
      </div>
    </div>

    <div class="card transport">
      <button class="icon play" id="rp-play" aria-label="재생/일시정지"></button>
      <div class="scrub-wrap">
        <div class="track">
          <input type="range" class="scrub" id="rp-scrub" min="0" max="100" value="0" step="1" aria-label="타임라인">
          <div class="ticks" id="rp-ticks"></div>
        </div>
        <div class="time"><span>t = <b id="rp-t2">0.0</b> s</span><span id="rp-frac"></span></div>
      </div>
      <div class="rt">
        <label class="toggle" id="rp-fov-l"><input type="checkbox" id="rp-fov">시야</label>
        <label class="toggle on" id="rp-trail-l"><input type="checkbox" id="rp-trail" checked>궤적</label>
        <div class="speed" id="rp-speed"></div>
      </div>
    </div>
    <div class="foot" id="rp-foot"></div>
  </div>
</div>

<script>
(function(){
  "use strict";
  var DATA = /*__DATA__*/null;
  if(!DATA){ return; }
  var $=function(id){return document.getElementById(id);};
  var M=DATA.meta, OBJS=DATA.objects, FR=DATA.frames, OD=DATA.obj_deltas,
      EV=DATA.events, AW=M.arena[0], AH=M.arena[1], G=M.geom;
  var N=FR.length;

  // ---- 라벨/색 사전 ----
  var CLS_KO={cube:"정육면체",octa:"정팔면체",dodeca:"정십이면체",icosa:"정이십면체",
    apple:"사과",orange:"오렌지",banana:"바나나",pineapple:"파인애플"};
  var SET_KO={set1:"세트1·다면체",set2:"세트2·과일큐브"};
  var STATE_KO=["대기","탐색 주행","후보 접근","정렬 접근","포획(블라인드)","후진 이탈",
    "운반","털어내기","밀어넣기","분리 후진","정차","완료","재정렬 후진"];
  // 상태 인덱스 -> 의미색 그룹
  var css=function(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim();};
  function stateColor(st){
    if(st===1||st===2) return css('--muted');           // 탐색/이동
    if(st===3||st===4) return css('--accent');           // 정렬/포획
    if(st===5) return css('--bad');                      // 이탈
    if(st===6) return css('--carry');                    // 운반
    if(st>=7&&st<=9) return css('--good');               // 하역
    if(st===12) return css('--carry');                   // 재정렬 후진
    if(st===11) return css('--good');                    // 완료
    return css('--muted2');
  }
  function objColor(o){
    var k = o.set==='set2' ? o.cls : o.cls;
    return css('--c-'+o.cls) || css('--muted');
  }

  // ---- 물체 상태 재구성 (델타 누적) ----
  // objState[i] = {x,y,s}  (mm, status)
  function buildAt(idx){
    var st=[];
    for(var i=0;i<OBJS.length;i++){ st.push({x:DATA.obj0[i][0],y:DATA.obj0[i][1],s:0}); }
    for(var f=0; f<=idx; f++){
      var d=OD[f]; if(!d) continue;
      for(var j=0;j<d.length;j++){ var e=d[j]; st[e[0]].x=e[1]; st[e[0]].y=e[2]; st[e[0]].s=e[3]; }
    }
    return st;
  }

  // ---- 캔버스/좌표 ----
  var cv=$('rp-cv'), ctx=cv.getContext('2d');
  var PAD=26, DPR=Math.max(1, Math.min(2.5, window.devicePixelRatio||1));
  var W=0, H=0, scale=1;
  function resize(){
    var cssW=cv.clientWidth||640; var cssH=cssW; // 정사각
    cv.style.height=cssH+'px';
    cv.width=Math.round(cssW*DPR); cv.height=Math.round(cssH*DPR);
    W=cssW; H=cssH; scale=(cssW-2*PAD)/AW;
    draw();
  }
  function SX(x){return PAD + x*scale;}
  function SY(y){return PAD + (AH - y)*scale;}   // y 뒤집기(위가 +y)

  // ---- 렌더 프리미티브 ----
  function rect(r,fill,stroke,lw){
    var x=SX(r[0]), y=SY(r[3]), w=(r[2]-r[0])*scale, h=(r[3]-r[1])*scale;
    if(fill){ctx.fillStyle=fill; ctx.fillRect(x,y,w,h);}
    if(stroke){ctx.strokeStyle=stroke; ctx.lineWidth=lw||1; ctx.strokeRect(x,y,w,h);}
  }
  function polyMarker(cx,cy,rad,sides,rot,fill,stroke,lw){
    ctx.beginPath();
    for(var i=0;i<sides;i++){
      var a=rot + i*2*Math.PI/sides;
      var px=cx+rad*Math.cos(a), py=cy+rad*Math.sin(a);
      i?ctx.lineTo(px,py):ctx.moveTo(px,py);
    }
    ctx.closePath();
    if(fill){ctx.fillStyle=fill; ctx.fill();}
    if(stroke){ctx.strokeStyle=stroke; ctx.lineWidth=lw||1.2; ctx.stroke();}
  }
  var SHAPE_SIDES={cube:4,octa:4,dodeca:5,icosa:6};   // 마커 변의 수(형상 힌트)
  var SHAPE_ROT={cube:Math.PI/4,octa:0,dodeca:-Math.PI/2,icosa:0};

  function drawObject(o, st){
    var cx=SX(st.x/1000), cy=SY(st.y/1000), col=objColor(o);
    var r=0.058*scale;
    if(st.s===3){                                  // 보관함 안착
      ctx.globalAlpha=0.9;
      if(o.set==='set2'){ ctx.beginPath(); ctx.arc(cx,cy,r*0.82,0,7); ctx.fillStyle=col; ctx.fill(); }
      else polyMarker(cx,cy,r*0.9,SHAPE_SIDES[o.cls],SHAPE_ROT[o.cls],col,null,0);
      // 안착 체크 테두리
      ctx.beginPath(); ctx.arc(cx,cy,r*1.15,0,7); ctx.strokeStyle=css('--good'); ctx.lineWidth=1.4; ctx.stroke();
      ctx.globalAlpha=1; return;
    }
    // 목표 강조 링(맥동)
    if(o.target){
      var pulse=reduce?0.5:(0.5+0.5*Math.sin(clock/380));
      ctx.beginPath(); ctx.arc(cx,cy,r+3+2*pulse,0,7);
      ctx.strokeStyle=css('--accent'); ctx.globalAlpha=0.35+0.45*pulse; ctx.lineWidth=1.6; ctx.stroke();
      ctx.globalAlpha=1;
    }
    if(o.set==='set2'){
      ctx.beginPath(); ctx.arc(cx,cy,r,0,7); ctx.fillStyle=col; ctx.fill();
      ctx.lineWidth=1; ctx.strokeStyle='rgba(0,0,0,.28)'; ctx.stroke();
    } else {
      polyMarker(cx,cy,r,SHAPE_SIDES[o.cls],SHAPE_ROT[o.cls],col,'rgba(0,0,0,.28)',1);
    }
    if(st.s===2){ // 입구에 문 물체 표시
      ctx.beginPath(); ctx.arc(cx,cy,r+2.5,0,7); ctx.strokeStyle=css('--carry');
      ctx.setLineDash([2,2]); ctx.lineWidth=1.4; ctx.stroke(); ctx.setLineDash([]);
    }
  }

  function drawRobot(f, fnext, a){
    var x=lerp(f[0],fnext[0],a)/1000, y=lerp(f[1],fnext[1],a)/1000;
    var yaw=lerpAng(f[2]/1000, fnext[2]/1000, a);
    var cx=SX(x), cy=SY(y), st=f[3];
    // 화면 좌표계(y下향)에서의 헤딩각
    var sa=-yaw;
    var col=css('--accent');
    // 카메라 시야
    if(showFOV){
      fovWedge(cx,cy,sa+Math.PI/2, G.search_fov, G.search_r*scale, col, 0.05);
      fovWedge(cx,cy,sa-Math.PI/2, G.search_fov, G.search_r*scale, col, 0.05);
      fovWedge(cx,cy,sa, G.verify_fov, G.verify_r*scale, css('--carry'), 0.09);
    }
    // ---- 실측 형상(robot.stl): 34×31cm 사각 섀시 + 전방 U자 스쿱 ----
    var S=scale;
    var fr=(G.rc_front||0.165)*S, bk=(G.rc_back||0.175)*S, hw=(G.width/2)*S;
    var ww=(G.wheel_w/2)*S, sh=(G.scoop_half||0.075)*S, sd=(G.scoop_depth||0.09)*S;
    var pk=G.pocket*S;
    ctx.save();
    ctx.translate(cx,cy);
    ctx.rotate(sa);                   // 로컬 +x=전방(헤딩), +y=측면
    // 구동바퀴 (차축=회전중심, 폭 밖으로 살짝)
    ctx.fillStyle=css('--muted2');
    ctx.fillRect(-0.05*S, hw, 0.10*S, (ww-hw)+2);
    ctx.fillRect(-0.05*S, -ww-2, 0.10*S, (ww-hw)+2);
    // 섀시 외곽: 전방 중앙에 U자 스쿱 컷
    ctx.beginPath();
    ctx.moveTo(-bk, -hw);
    ctx.lineTo(fr, -hw);
    ctx.lineTo(fr, -sh);              // 스쿱 우측 입구
    ctx.lineTo(fr-sd, -sh);           // 안쪽(닫힌 끝)
    ctx.lineTo(fr-sd, sh);
    ctx.lineTo(fr, sh);               // 스쿱 좌측 입구
    ctx.lineTo(fr, hw);
    ctx.lineTo(-bk, hw);
    ctx.closePath();
    ctx.fillStyle=css('--panel2'); ctx.globalAlpha=0.96; ctx.fill(); ctx.globalAlpha=1;
    ctx.lineWidth=2; ctx.strokeStyle=col; ctx.stroke();
    // 스쿱 포켓(물체 안착점)
    ctx.beginPath(); ctx.arc(pk,0,0.02*S,0,7);
    ctx.strokeStyle=col; ctx.globalAlpha=0.7; ctx.lineWidth=1; ctx.stroke(); ctx.globalAlpha=1;
    // 라이다(전방중앙) · 카메라 마스트(중심)
    ctx.beginPath(); ctx.arc(0.06*S,0,0.028*S,0,7);
    ctx.fillStyle=css('--c-dodeca'); ctx.globalAlpha=0.85; ctx.fill(); ctx.globalAlpha=1;
    ctx.beginPath(); ctx.arc(-0.03*S,0,0.024*S,0,7); ctx.fillStyle=css('--ink'); ctx.fill();
    // IR 적재 표시(포켓에 점)
    if(f[4]){ ctx.beginPath(); ctx.arc(pk,0,0.03*S,0,7); ctx.fillStyle=css('--carry'); ctx.fill(); }
    ctx.restore();
    return {cx:cx,cy:cy,x:x,y:y,tgt:f[7]};
  }
  function fovWedge(cx,cy,dir,fovDeg,rad,color,alpha){
    var h=fovDeg*Math.PI/180/2;
    ctx.beginPath(); ctx.moveTo(cx,cy);
    ctx.arc(cx,cy,rad,dir-h,dir+h); ctx.closePath();
    ctx.fillStyle=color; ctx.globalAlpha=alpha; ctx.fill(); ctx.globalAlpha=1;
  }

  function drawGrid(){
    ctx.fillStyle=css('--bg'); ctx.fillRect(0,0,W,H);
    // 경기장 바닥
    rect([0,0,AW,AH], css('--panel'), null,0);
    // 격자 0.5m
    ctx.strokeStyle=css('--grid'); ctx.lineWidth=1;
    for(var gx=0; gx<=AW+0.001; gx+=0.5){ ctx.beginPath(); ctx.moveTo(SX(gx),SY(0)); ctx.lineTo(SX(gx),SY(AH)); ctx.stroke(); }
    for(var gy=0; gy<=AH+0.001; gy+=0.5){ ctx.beginPath(); ctx.moveTo(SX(0),SY(gy)); ctx.lineTo(SX(AW),SY(gy)); ctx.stroke(); }
    // 스타트존 / 보관함
    rect(M.start_zone, 'color-mix(in srgb,'+css('--accent')+' 9%, transparent)', css('--accent'),1.3);
    rect(M.sticker_zone, 'color-mix(in srgb,'+css('--good')+' 6%, transparent)', null,0);
    rect(M.score_box, 'color-mix(in srgb,'+css('--good')+' 12%, transparent)', css('--good'),1.4);
    // 라벨
    ctx.font='10px '+css('--mono'); ctx.textBaseline='alphabetic';
    ctx.fillStyle=css('--muted');
    ctx.textAlign='right'; ctx.fillText('스타트존', SX(M.start_zone[2])-4, SY(M.start_zone[3])-5);
    ctx.textAlign='left'; ctx.fillStyle=css('--good');
    ctx.fillText('보관함', SX(M.score_box[0])+3, SY(M.score_box[3])-5);
    // 외곽
    ctx.strokeStyle=css('--line'); ctx.lineWidth=1.5; rect([0,0,AW,AH], null, css('--line'),1.5);
    // 눈금(모서리 m)
    ctx.fillStyle=css('--muted2'); ctx.font='9px '+css('--mono');
    ctx.textAlign='center'; ctx.fillText('0', SX(0), SY(0)+13);
    ctx.fillText(AW+'m', SX(AW), SY(0)+13);
    ctx.textAlign='right'; ctx.fillText(AH+'m', SX(0)-4, SY(AH)+3);
  }

  function drawTrail(upto,curx,cury){
    if(!showTrail) return;
    ctx.lineWidth=2; ctx.lineJoin='round'; ctx.lineCap='round';
    ctx.beginPath();
    var started=false, step=Math.max(1,Math.floor(N/900));
    for(var i=0;i<=upto;i+=step){
      var px=SX(FR[i][0]/1000), py=SY(FR[i][1]/1000);
      if(!started){ctx.moveTo(px,py); started=true;} else ctx.lineTo(px,py);
    }
    ctx.lineTo(SX(curx),SY(cury));
    ctx.strokeStyle=css('--accent'); ctx.globalAlpha=0.45; ctx.stroke(); ctx.globalAlpha=1;
  }

  // ---- 보간 유틸 ----
  function lerp(a,b,t){return a+(b-a)*t;}
  function lerpAng(a,b,t){
    var d=b-a; while(d>Math.PI)d-=2*Math.PI; while(d<-Math.PI)d+=2*Math.PI;
    return a+d*t;
  }

  // ---- 메인 draw ----
  var reduce=window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var showFOV=false, showTrail=true, clock=0;
  function draw(){
    ctx.save(); ctx.scale(DPR,DPR);
    drawGrid();
    var i=Math.min(N-1, Math.floor(cur)); var a=cur-i; var inext=Math.min(N-1,i+1);
    var f=FR[i], fn=FR[inext];
    var objs=buildAt(i);
    // 궤적
    var rx=lerp(f[0],fn[0],a)/1000, ry=lerp(f[1],fn[1],a)/1000;
    drawTrail(i,rx,ry);
    // 목표 추적선
    var tgtId=f[7];
    if(tgtId>=0){
      for(var k=0;k<OBJS.length;k++) if(OBJS[k].id===tgtId){
        var ox=objs[k].x/1000, oy=objs[k].y/1000;
        ctx.setLineDash([4,4]); ctx.strokeStyle=css('--accent'); ctx.globalAlpha=0.5; ctx.lineWidth=1.3;
        ctx.beginPath(); ctx.moveTo(SX(rx),SY(ry)); ctx.lineTo(SX(ox),SY(oy)); ctx.stroke();
        ctx.setLineDash([]); ctx.globalAlpha=1; break;
      }
    }
    // 물체(안착 먼저 깔고, 나머지 위에)
    for(var p=0;p<OBJS.length;p++) if(objs[p].s===3) drawObject(OBJS[p],objs[p]);
    for(var q=0;q<OBJS.length;q++) if(objs[q].s!==3 && objs[q].s!==1) drawObject(OBJS[q],objs[q]);
    // 로봇
    var rob=drawRobot(f,fn,a);
    // 적재물(로봇 위에)
    for(var s=0;s<OBJS.length;s++) if(objs[s].s===1) drawObject(OBJS[s],objs[s]);
    ctx.restore();
    updateHUD(i,f);
  }

  // ---- HUD ----
  var chip=$('rp-state'), logEl=$('rp-log'), lastLogIdx=-1;
  function updateHUD(i,f){
    var st=f[3];
    chip.textContent=STATE_KO[st]; chip.style.setProperty('--sc', stateColor(st));
    $('rp-t').textContent=(f[8]/10).toFixed(1);
    $('rp-t2').textContent=(f[8]/10).toFixed(1);
    // 진행 점수/하역: 지금까지 안착한 목표/비목표 집계
    var objs=buildAt(i), good=0,bad=0,pts=0;
    for(var k=0;k<OBJS.length;k++){
      if(objs[k].s===3){
        var o=OBJS[k], v=o.set==='set2'?20:10;
        if(o.target){good++; pts+=v;} else {bad++; pts-=2*v;}
      }
    }
    $('rp-pts').textContent=(pts>0?'+':'')+pts;
    $('rp-pts').className='v'+(pts>0?' good':(pts<0?' bad':''));
    $('rp-good').textContent=good;
    $('rp-bad').textContent=bad+' / '+wallHitsUpto(f[8]);
    // 적재 상태
    var carry=$('rp-carry'), ct=$('rp-carry-t');
    if(f[5]>=0){ carry.className='carrying'; var oo=objById(f[5]);
      ct.textContent='운반 중: '+CLS_KO[oo.cls]+(f[6]>=0?' (+입구 1)':''); }
    else { carry.className='carrying off'; ct.textContent='빈 비어있음'; }
    // 로그 하이라이트
    highlightLog(i);
  }
  function wallHitsUpto(ds){ var c=0; for(var e=0;e<EV.length;e++){ if(EV[e].t*10<=ds && /WALL/i.test(EV[e].txt)) c++; } return M.result.wall_hits; }
  function objById(id){ for(var k=0;k<OBJS.length;k++) if(OBJS[k].id===id) return OBJS[k]; return {cls:'?'}; }

  // ---- 이벤트 로그 ----
  var EV_KO=[
    [/PAYLOAD_LOST/i,'적재물 유실 감지'],[/CAPTURE_MISSED/i,'포획 빗맞음 → 재시도'],
    [/CAPTURE_READY/i,'정렬 완료 · 포획 개시'],[/DEPOSIT|STORED|GOOD_DROP/i,'보관함 하역'],
    [/MISPICK|BAD/i,'오픽업(감점)'],[/WALL/i,'벽 접촉'],[/CONFIRM/i,'목표 확정'],
    [/RETREAT|ABORT/i,'이탈/중단'],[/SHED/i,'밀항 물체 털어내기'],[/TIME/i,'시간 컷오프']
  ];
  function evLabel(txt){ for(var i=0;i<EV_KO.length;i++) if(EV_KO[i][0].test(txt)) return EV_KO[i][1]; return txt; }
  function buildLog(){
    logEl.innerHTML='';
    for(var i=0;i<EV.length;i++){
      var d=document.createElement('div'); d.className='ev'; d.dataset.fi=EV[i].fi;
      d.innerHTML='<span class="tt">'+EV[i].t.toFixed(1)+'s</span><span>'+evLabel(EV[i].txt)+'</span>';
      logEl.appendChild(d);
    }
    if(!EV.length){ logEl.innerHTML='<div class="ev"><span class="tt">—</span><span>기록된 이벤트 없음(무사 완주)</span></div>'; }
  }
  function highlightLog(i){
    var rows=logEl.querySelectorAll('.ev'), latest=-1;
    for(var r=0;r<rows.length;r++){ var fi=+rows[r].dataset.fi;
      rows[r].classList.toggle('now', fi<=i && (r+1>=rows.length || +rows[r+1].dataset.fi>i));
      if(fi<=i) latest=r;
    }
  }

  // ---- 재생 엔진 ----
  var cur=0, playing=false, speed=1, lastTs=null;
  var scrub=$('rp-scrub'); scrub.max=N-1;
  function frame(ts){
    if(playing){
      if(lastTs!=null){
        var dtms=ts-lastTs;
        // 기록 프레임 간격 = rec_every*dt 초; 실시간 배속 speed
        var fps=1000*(M.dt*M.rec_every); // ms per frame
        cur += (dtms/fps)*speed;
        if(cur>=N-1){ cur=N-1; setPlaying(false); }
      }
      lastTs=ts; scrub.value=Math.floor(cur);
    }
    clock=ts;
    draw();
    requestAnimationFrame(frame);
  }
  function setPlaying(p){
    playing=p; lastTs=null;
    if(p && cur>=N-1) cur=0;
    $('rp-play').innerHTML = p
      ? '<svg viewBox="0 0 24 24"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>'
      : '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
    $('rp-play').classList.toggle('play', !p);
  }
  $('rp-play').onclick=function(){ setPlaying(!playing); };
  scrub.oninput=function(){ cur=+scrub.value; if(playing) setPlaying(false); };

  // 속도 버튼
  var SP=[0.5,1,2,4], spWrap=$('rp-speed');
  SP.forEach(function(s){ var b=document.createElement('button'); b.textContent=s+'×';
    if(s===1)b.className='on'; b.onclick=function(){ speed=s;
      spWrap.querySelectorAll('button').forEach(function(x){x.classList.remove('on');});
      b.classList.add('on'); }; spWrap.appendChild(b); });

  $('rp-fov').onchange=function(e){ showFOV=e.target.checked; $('rp-fov-l').classList.toggle('on',showFOV); };
  $('rp-trail').onchange=function(e){ showTrail=e.target.checked; $('rp-trail-l').classList.toggle('on',showTrail); };

  // ---- 타임라인 이벤트 틱 ----
  function buildTicks(){
    var t=$('rp-ticks'); t.innerHTML='';
    EV.forEach(function(e){ var d=document.createElement('div'); d.className='tick';
      d.style.left=(100*e.fi/(N-1))+'%';
      var c= /LOST|MISS|WALL|BAD|MISPICK|ABORT/i.test(e.txt)?css('--bad'):
             /DEPOSIT|STORED|GOOD/i.test(e.txt)?css('--good'):css('--accent');
      d.style.background=c; t.appendChild(d); });
  }

  // ---- 범례 ----
  function buildLegend(){
    var L=$('rp-legend'); var rows=[];
    var s1=['octa','dodeca','icosa','cube'].filter(function(c){return OBJS.some(function(o){return o.set==='set1'&&o.cls===c;});});
    var s2=['apple','orange','banana','pineapple'].filter(function(c){return OBJS.some(function(o){return o.set==='set2'&&o.cls===c;});});
    function swatch(o){ var col=css('--c-'+o); var sh=(['apple','orange','banana','pineapple'].indexOf(o)>=0)?'50%':'2px';
      return '<span class="swatch" style="background:'+col+';border-radius:'+sh+'"></span>'; }
    rows.push('<div class="lg-row" style="color:var(--ink);font-weight:600">'+SET_KO.set1+'</div>');
    s1.forEach(function(c){ rows.push('<div class="lg-row">'+swatch(c)+'<span>'+CLS_KO[c]+(M.targets.set1===c?' <b>· 목표</b>':'')+'</span></div>'); });
    rows.push('<div class="lg-row" style="color:var(--ink);font-weight:600;margin-top:4px">'+SET_KO.set2+'</div>');
    s2.forEach(function(c){ rows.push('<div class="lg-row">'+swatch(c)+'<span>'+CLS_KO[c]+(M.targets.set2===c?' <b>· 목표</b>':'')+'</span></div>'); });
    rows.push('<div class="lg-row" style="margin-top:6px"><span class="swatch" style="border:2px solid var(--accent);border-radius:50%;background:transparent"></span><span>로봇 · <b>목표 링</b>/궤적</span></div>');
    rows.push('<div class="lg-row"><span class="swatch" style="border:1.5px solid var(--good);border-radius:50%;background:transparent"></span><span>보관함 안착</span></div>');
    L.innerHTML=rows.join('');
  }

  // ---- 헤더/메타 ----
  function fillMeta(){
    var r=M.result;
    $('rp-title').textContent='경기 리플레이 · seed '+M.seed+' · '+M.mode+' 모드';
    document.title='경기 리플레이 · seed '+M.seed;
    var tg=[];
    if(M.targets.set1) tg.push('세트1 '+CLS_KO[M.targets.set1]);
    if(M.targets.set2) tg.push('세트2 '+CLS_KO[M.targets.set2]);
    $('rp-meta').innerHTML=
      '<span>목표 <b>'+tg.join(' · ')+'</b></span>'+
      '<span>순항 <b>'+M.cruise+' m/s</b></span>'+
      '<span>결과 <b>'+(r.points>0?'+':'')+r.points+'점</b> · 하역 '+r.good+' · 오픽업 '+r.bad+'</span>'+
      '<span>종료 <b>'+(r.t_end).toFixed(0)+'s / '+M.duration.toFixed(0)+'s</b></span>';
    $('rp-tend').textContent=' / '+M.duration.toFixed(0)+'s';
    $('rp-foot').textContent='navigation/replay_mission.py 로 생성 · 실제 mission_fsm 로직을 그대로 구동한 3분 경기 기록';
  }

  // ---- 부트 ----
  fillMeta(); buildLegend(); buildLog(); buildTicks(); setPlaying(false);
  window.addEventListener('resize', resize);
  new ResizeObserver(resize).observe(cv);
  resize();
  requestAnimationFrame(frame);
  // 자동 재생 (모션 축소 아니면)
  if(!reduce) setTimeout(function(){ setPlaying(true); }, 500);
})();
</script>
"""


# ----------------------------------------------------------------------- CLI

def _summary_line(seed, res):
    return (f"  seed {seed:2d}: {res['points']:+5.0f}점  하역 {res['good']} "
            f"오픽업 {res['bad']}  벽 {res['wall_hits']}  "
            f"하역시각 {res['deposit_times']}  종료 {res['end_state']} "
            f"@{res['t_end']:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--mode", default="both",
                    choices=["both", "bothcube", "set1", "set1cube", "set2"])
    ap.add_argument("--cruise", type=float, default=0.20)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--rec-every", type=int, default=2,
                    help="몇 틱마다 기록할지 (2 = 10Hz)")
    ap.add_argument("--out", default=None, help="HTML 출력 경로")
    ap.add_argument("--fragment", action="store_true",
                    help="doctype/head 없는 본문만 (아티팩트용)")
    ap.add_argument("--json", action="store_true", help="궤적 JSON 을 stdout 으로")
    ap.add_argument("--scan", nargs=2, type=int, metavar=("LO", "HI"),
                    help="[LO,HI) 시드 요약만 출력해 좋은 경기 고르기")
    args = ap.parse_args()

    if args.scan:
        lo, hi = args.scan
        print(f"[scan {args.mode} @ cruise {args.cruise}]")
        for s in range(lo, hi):
            _, res = record(s, args.mode, args.cruise, args.duration, args.rec_every)
            print(_summary_line(s, res))
        return

    data, res = record(args.seed, args.mode, args.cruise, args.duration,
                        args.rec_every)
    if args.json:
        print(json.dumps(data, ensure_ascii=False))
        return

    html = render_html(data, fragment=args.fragment)
    out = args.out or f"navigation/replays/match_seed{args.seed}_{args.mode}.html"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(_summary_line(args.seed, res))
    print(f"  프레임 {len(data['frames'])}개, 이벤트 {len(data['events'])}개")
    print(f"  → {out}  ({os.path.getsize(out)//1024} KB)  브라우저로 여세요")


if __name__ == "__main__":
    main()
