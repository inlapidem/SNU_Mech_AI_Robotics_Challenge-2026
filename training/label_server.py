"""Zero-install browser box-labeler for the staged Set-1 detector frames.

No GUI toolkit needed: run it, open http://localhost:8765 in any browser (works from WSL via
localhost forwarding). Each frame shows the pre-filled box; drag to move, drag a corner to
resize, draw on empty space to add, press D / "Delete box" to remove, "No object" to clear.
Right arrow / Space saves and advances; edits are written straight back to the .txt files.

    yolo/bin/python training/label_server.py
Then: yolo/bin/python training/autolabel_detector.py --mode finalize
"""
import glob
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE = os.path.join(ROOT, "datasets", "set1_autolabel", "stage")
REVIEWED = os.path.join(STAGE, "_reviewed.json")
FOLDERS = ["cube", "dodeca", "icosa", "octa"]
PORT = 8765


def list_items():
    items = []
    for c in FOLDERS:
        for png in sorted(glob.glob(os.path.join(STAGE, c, "*.png"))):
            stem = os.path.splitext(os.path.basename(png))[0]
            txt = os.path.join(STAGE, c, stem + ".txt")
            boxes = []
            if os.path.isfile(txt):
                for ln in open(txt).read().splitlines():
                    p = ln.split()
                    if len(p) == 5:
                        boxes.append([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
            items.append({"cls": c, "stem": stem, "url": f"/img/{c}/{stem}.png", "boxes": boxes})
    return items


def load_reviewed():
    if os.path.isfile(REVIEWED):
        try:
            return set(json.load(open(REVIEWED)))
        except Exception:
            return set()
    return set()


def save_reviewed(s):
    json.dump(sorted(s), open(REVIEWED, "w"))


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>set1 labeler</title>
<style>
 body{margin:0;font-family:system-ui,sans-serif;background:#1e1e1e;color:#ddd}
 #bar{padding:8px 12px;background:#2d2d2d;position:sticky;top:0;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 #bar b{color:#4ec9b0} button{background:#3a3d41;color:#ddd;border:1px solid #555;padding:6px 10px;border-radius:4px;cursor:pointer}
 button:hover{background:#4a4d51} #wrap{padding:12px;text-align:center} canvas{background:#000;cursor:crosshair;max-width:100%}
 .k{color:#888;font-size:12px} #done{color:#c586c0}
</style></head><body>
<div id=bar>
 <span id=pos></span> <span id=cls></span> <span id=done></span>
 <button onclick=prev()>&larr; Prev</button>
 <button onclick=save(true)>Save + Next &rarr;</button>
 <button onclick=delSel()>Delete box (D)</button>
 <button onclick=clearAll()>No object</button>
 <span class=k>drag=move · corner=resize · drag empty=new box · &larr;/&rarr;/Space=nav</span>
</div>
<div id=wrap><canvas id=cv></canvas></div>
<script>
let items=[],i=0,natW=0,natH=0,sc=1,boxes=[],sel=-1,mode=null,corner=-1,start=null,img=new Image();
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const MAXW=1100;
function nkey(it){return it.cls+'/'+it.stem}
async function boot(){items=await (await fetch('/api/list')).json();document.getElementById('done').innerHTML='';load(0);}
function load(n){if(n<0||n>=items.length)return;i=n;const it=items[i];
 document.getElementById('pos').textContent=(i+1)+' / '+items.length;
 document.getElementById('cls').innerHTML='<b>'+it.cls+'</b> '+it.stem;
 document.getElementById('done').innerHTML=it.done?'<span id=done>reviewed ✔</span>':'';
 img.onload=()=>{natW=img.naturalWidth;natH=img.naturalHeight;sc=Math.min(1,MAXW/natW);
  cv.width=natW*sc;cv.height=natH*sc;
  boxes=it.boxes.map(b=>({x0:(b[0]-b[2]/2)*natW,y0:(b[1]-b[3]/2)*natH,x1:(b[0]+b[2]/2)*natW,y1:(b[1]+b[3]/2)*natH}));
  sel=boxes.length?0:-1;draw();};
 img.src=it.url;}
function draw(){ctx.clearRect(0,0,cv.width,cv.height);ctx.drawImage(img,0,0,cv.width,cv.height);
 boxes.forEach((b,j)=>{ctx.lineWidth=j===sel?3:2;ctx.strokeStyle=j===sel?'#4ec9b0':'#f0c040';
  ctx.strokeRect(b.x0*sc,b.y0*sc,(b.x1-b.x0)*sc,(b.y1-b.y0)*sc);
  if(j===sel){ctx.fillStyle='#4ec9b0';for(const[cx,cy]of corners(b))ctx.fillRect(cx*sc-4,cy*sc-4,8,8);}});}
function corners(b){return[[b.x0,b.y0],[b.x1,b.y0],[b.x0,b.y1],[b.x1,b.y1]];}
function pos(e){const r=cv.getBoundingClientRect();return{x:(e.clientX-r.left)/sc,y:(e.clientY-r.top)/sc};}
cv.onmousedown=e=>{const p=pos(e);
 if(sel>=0){const cs=corners(boxes[sel]);for(let k=0;k<4;k++)if(Math.abs(p.x-cs[k][0])<8&&Math.abs(p.y-cs[k][1])<8){mode='resize';corner=k;return;}}
 for(let j=0;j<boxes.length;j++){const b=boxes[j];if(p.x>=b.x0&&p.x<=b.x1&&p.y>=b.y0&&p.y<=b.y1){sel=j;mode='move';start={x:p.x,y:p.y};draw();return;}}
 sel=boxes.length;boxes.push({x0:p.x,y0:p.y,x1:p.x,y1:p.y});mode='new';draw();};
cv.onmousemove=e=>{if(!mode)return;const p=pos(e),b=boxes[sel];
 if(mode==='move'){const dx=p.x-start.x,dy=p.y-start.y;b.x0+=dx;b.x1+=dx;b.y0+=dy;b.y1+=dy;start={x:p.x,y:p.y};}
 else if(mode==='new'){b.x1=p.x;b.y1=p.y;}
 else if(mode==='resize'){if(corner==0){b.x0=p.x;b.y0=p.y;}if(corner==1){b.x1=p.x;b.y0=p.y;}if(corner==2){b.x0=p.x;b.y1=p.y;}if(corner==3){b.x1=p.x;b.y1=p.y;}}
 draw();};
cv.onmouseup=()=>{if(mode){const b=boxes[sel];const nb={x0:Math.min(b.x0,b.x1),y0:Math.min(b.y0,b.y1),x1:Math.max(b.x0,b.x1),y1:Math.max(b.y0,b.y1)};
 boxes[sel]=nb;if((nb.x1-nb.x0)<6||(nb.y1-nb.y0)<6){boxes.splice(sel,1);sel=boxes.length-1;}draw();}mode=null;corner=-1;};
function delSel(){if(sel>=0){boxes.splice(sel,1);sel=boxes.length-1;draw();}}
function clearAll(){boxes=[];sel=-1;draw();}
async function save(next){const it=items[i];
 const nb=boxes.filter(b=>b.x1-b.x0>3&&b.y1-b.y0>3).map(b=>[(b.x0+b.x1)/2/natW,(b.y0+b.y1)/2/natH,(b.x1-b.x0)/natW,(b.y1-b.y0)/natH]);
 it.boxes=nb;it.done=true;
 await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cls:it.cls,stem:it.stem,boxes:nb})});
 if(next)load(Math.min(i+1,items.length-1));else draw();}
function prev(){save(false).then(()=>load(i-1));}
document.onkeydown=e=>{if(e.key==='ArrowRight'||e.key===' '){e.preventDefault();save(true);}
 else if(e.key==='ArrowLeft'){e.preventDefault();save(false).then(()=>load(i-1));}
 else if(e.key==='d'||e.key==='Delete'||e.key==='Backspace'){delSel();}};
boot();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/api/list":
            reviewed = load_reviewed()
            items = list_items()
            for it in items:
                it["done"] = nkey(it) in reviewed
            return self._send(200, json.dumps(items))
        m = re.match(r"^/img/([^/]+)/([^/?]+)$", self.path)
        if m:
            fp = os.path.join(STAGE, m.group(1), m.group(2))
            if os.path.isfile(fp):
                return self._send(200, open(fp, "rb").read(), "image/png")
        self._send(404, "{}")

    def do_POST(self):
        if self.path != "/api/save":
            return self._send(404, "{}")
        n = int(self.headers.get("Content-Length", 0))
        d = json.loads(self.rfile.read(n) or b"{}")
        txt = os.path.join(STAGE, d["cls"], d["stem"] + ".txt")
        with open(txt, "w") as f:
            for b in d.get("boxes", []):
                f.write(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
        r = load_reviewed(); r.add(d["cls"] + "/" + d["stem"]); save_reviewed(r)
        self._send(200, json.dumps({"ok": True}))

    def log_message(self, *a):
        pass


def nkey(it):
    return it["cls"] + "/" + it["stem"]


if __name__ == "__main__":
    n = len(list_items())
    print(f"labeler: {n} staged frames.  open  http://localhost:{PORT}")
    print("edits save to the .txt files; then run  autolabel_detector.py --mode finalize")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
