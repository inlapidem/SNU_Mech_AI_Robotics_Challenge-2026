#!/usr/bin/env python3
"""
맵 웹서버 (정적 파일 서빙 + 초기화 API).

기존의 `python3 -m http.server 8000` 을 대체한다. maps/ 디렉토리를 그대로 서빙하되,
브라우저의 '지도 초기화' 버튼이 보내는  POST /api/reset  을 받아 reset_map.sh 를 실행
(=slam_toolbox 재시작 → 누적 맵 삭제 후 처음부터)한다.

사용:  python3 map_server.py [port]     (기본 8000)
"""
import os
import sys
import subprocess
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.expanduser('~/lidar_ws/maps')
ACTIONS = {
    '/api/reset':        os.path.expanduser('~/lidar_ws/reset_map.sh'),     # 지도 초기화(slam 재시작)
    '/api/reset_origin': os.path.expanduser('~/lidar_ws/reset_origin.sh'),  # 위치 원점 리셋
}
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def _json(self, code, body):
        data = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        script = ACTIONS.get(self.path.rstrip('/'))
        if script:
            try:
                subprocess.Popen(['bash', script],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL,
                                 start_new_session=True)
                self._json(200, '{"ok":true,"msg":"started"}')
            except Exception as e:
                self._json(500, '{"ok":false,"msg":"%s"}' % str(e).replace('"', "'"))
        else:
            self._json(404, '{"ok":false,"msg":"unknown endpoint"}')

    def log_message(self, *a):
        pass  # 조용히


if __name__ == '__main__':
    print(f'map_server: http://0.0.0.0:{PORT}  (root={ROOT}, POST /api/reset -> reset_map.sh)')
    ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
