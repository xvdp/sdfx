"""Minimal PNG writer (8-bit RGBA, no dependencies beyond NumPy), shared by the Python hosts."""

import struct
import zlib

import numpy as np


def write_png(path: str, rgba: np.ndarray) -> None:
	"""(h, w, 4) uint8, first row = top of the image."""
	h, w = rgba.shape[:2]
	raw = b''.join(b'\x00' + np.ascontiguousarray(rgba[y], np.uint8).tobytes() for y in range(h))

	def chunk(tag, data):
		return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)

	with open(path, 'wb') as f:
		f.write(b'\x89PNG\r\n\x1a\n')
		f.write(chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)))
		f.write(chunk(b'IDAT', zlib.compress(raw, 6)))
		f.write(chunk(b'IEND', b''))
