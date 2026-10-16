"""
Local HTTP server for the browser hosts: serves the sdfview package directory (web/ pages and modules, shaders/ code)
and scene files under /scenes/<name>. Browsers only expose WebGPU to secure contexts; http://localhost is one.

	python -m sdfview serve tooth.sdfx knot.sdfx [--port 8000]
	-> http://localhost:8000/web/webgl2/?scene=/scenes/tooth.sdfx   (and webgpu/, three/webgl.html, three/webgpu.html)
"""

from __future__ import annotations

import functools
import http.server
import os
import threading
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))

HOST_PAGES = {
	'webgl2': '/web/webgl2/',
	'webgpu': '/web/webgpu/',
	'three-webgl': '/web/three/webgl.html',
	'three-webgpu': '/web/three/webgpu.html',
	'three-tsl-webgl': '/web/three/webgpu.html?backend=webgl',
	'three-scene': '/web/three/compose.html',
}


class Handler(http.server.SimpleHTTPRequestHandler):
	scenes: Dict[str, str] = {}
	extensions_map = {
		**http.server.SimpleHTTPRequestHandler.extensions_map,
		'.js': 'text/javascript',
		'.mjs': 'text/javascript',
		'.wgsl': 'text/plain',
		'.glsl': 'text/plain',
		'.frag': 'text/plain',
		'.vert': 'text/plain',
		'.slang': 'text/plain',
		'.sdfx': 'application/octet-stream',
	}

	def translate_path(self, path):
		clean = path.split('?', 1)[0].split('#', 1)[0]
		if clean.startswith('/scenes/'):
			name = clean[len('/scenes/') :]
			if name in self.scenes:
				return self.scenes[name]
		return super().translate_path(path)

	def end_headers(self):
		self.send_header('Cache-Control', 'no-store')  # edited shaders are reloaded on refresh
		super().end_headers()

	def log_message(self, fmt, *args):  # quiet
		pass


def make_server(scenes: Optional[Dict[str, str]] = None, port: int = 8000, root: str = ROOT):
	"""ThreadingHTTPServer on localhost: `root` at /, each scenes[name] (a file path) at /scenes/<name>."""
	handler = type('SceneHandler', (Handler,), {'scenes': {k: os.path.abspath(v) for k, v in (scenes or {}).items()}})
	return http.server.ThreadingHTTPServer(('localhost', port), functools.partial(handler, directory=root))


def serve_in_thread(scenes: Optional[Dict[str, str]] = None, port: int = 0):
	"""Start a server in a daemon thread (port 0: any free port); returns (server, base_url)."""
	server = make_server(scenes, port)
	threading.Thread(target=server.serve_forever, daemon=True).start()
	return server, f'http://localhost:{server.server_address[1]}'


def serve(paths, port: int = 8000) -> None:
	scenes = {os.path.basename(p): p for p in paths}
	server = make_server(scenes, port)
	base = f'http://localhost:{server.server_address[1]}'
	print(f'serving {ROOT} at {base}')
	for name in scenes or ['<scene>.sdfx']:
		for host, page in HOST_PAGES.items():
			print(f'  {host:15s} {base}{page}{"&" if "?" in page else "?"}scene=/scenes/{name}')
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		pass
