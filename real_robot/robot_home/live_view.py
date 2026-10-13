#!/usr/bin/env python3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64

PORT = 8090
STALE_S = 1.0
FRONT_TO_CENTRE = 0.161 - 0.01235
_state = {'frame': None, 'src': 'waiting', 'seq': 0, 't_debug': 0.0, 'pose': None, 'age': None}
_lock = threading.Lock()

def mark_raw(img):
    img[:30] = (img[:30] * 0.3).astype(img.dtype)
    txt = 'CAMERA ONLY - detector not publishing'
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (6, 6), (18 + tw, 24), (0, 170, 255), -1)
    cv2.putText(img, txt, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img

class Viewer(Node):
    def __init__(self):
        super().__init__('live_view')
        self.bridge = CvBridge()
        self.create_subscription(Image, '/box_detection_debug', self._dbg, 5)
        self.create_subscription(Image, '/camera/color/image_raw', self._raw, 5)
        self.create_subscription(PoseStamped, '/box_pose', self._pose, 5)
        self.create_subscription(Float64, '/box_detection_age', self._age, 5)

    def _store(self, msg, src):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return
        now = time.time()
        with _lock:
            if src == 'debug':
                _state['t_debug'] = now
            elif now - _state['t_debug'] < STALE_S:
                return
        if src == 'raw':
            img = mark_raw(img)
        with _lock:
            _state['frame'], _state['src'] = img, src
            _state['seq'] += 1

    def _dbg(self, msg):
        self._store(msg, 'debug')

    def _raw(self, msg):
        self._store(msg, 'raw')

    def _pose(self, msg):
        p = msg.pose.position
        with _lock:
            _state['pose'] = (p.x, p.y, p.z)

    def _age(self, msg):
        with _lock:
            _state['age'] = msg.data

def reading():
    with _lock:
        p, age, src = _state['pose'], _state['age'], _state['src']
    if src == 'raw':
        return 'detector not publishing - camera only'
    if p is None:
        return 'no cube seen yet'
    txt = (f'chassis front -> cube centre {(p[0] - FRONT_TO_CENTRE) * 100:.1f} cm   |   '
           f'{abs(p[1]) * 100:.1f} cm {"LEFT" if p[1] > 0 else "RIGHT"} of robot centre')
    if age is not None and age > STALE_S:
        txt += f'   |   last seen {age:.0f} s ago' if age != float('inf') else '   |   not seen'
    return txt

PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>LIMO camera</title>
<style>
body{margin:0;padding:16px;background:#16191d;color:#d9dee4;font:15px/1.4 system-ui,sans-serif;
     display:flex;flex-direction:column;align-items:center;gap:10px}
h1{margin:0;font-size:14px;font-weight:600;letter-spacing:.08em;color:#8f99a3}
img{width:min(100%,800px);border-radius:4px;background:#000}
#p{font-size:17px;font-variant-numeric:tabular-nums}
</style></head>
<body><h1>LIMO CAMERA - LIVE</h1>
<img src="/stream" alt="robot camera"><div id="p">&nbsp;</div>
<script>setInterval(async()=>{try{const r=await fetch('/pose');
document.getElementById('p').textContent=await r.text();}catch(e){}},400);</script>
</body></html>"""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == '/pose':
            b = reading().encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        if self.path != '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
            return
        self.send_response(200)
        self.send_header('Content-Type',
                         'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        last = -1
        try:
            while True:
                with _lock:
                    seq, img = _state['seq'], _state['frame']
                if img is None or seq == last:
                    time.sleep(0.02)
                    continue
                last = seq
                ok, jpg = cv2.imencode('.jpg', img,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok:
                    continue
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n'
                                 b'Content-Length: ' + str(len(jpg)).encode() +
                                 b'\r\n\r\n' + jpg.tobytes() + b'\r\n')
        except (BrokenPipeError, ConnectionResetError):
            pass

def main():
    rclpy.init()
    node = Viewer()
    threading.Thread(
        target=lambda: ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever(),
        daemon=True).start()
    print(f'live view on http://0.0.0.0:{PORT}', flush=True)
    rclpy.spin(node)

if __name__ == '__main__':
    main()
