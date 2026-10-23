"""Runtime helpers around libedgetpu."""

import ctypes
from ctypes import c_char_p, c_int, c_size_t, POINTER, Structure

from .paths import require_runtime


class EdgetpuDevice(Structure):
    _fields_ = [("type", c_int), ("path", c_char_p)]


class CoralRuntime:
    def __init__(self):
        self.lib = ctypes.CDLL(require_runtime())
        self.lib.edgetpu_version.restype = c_char_p
        self.lib.edgetpu_verbosity.argtypes = [c_int]
        self.lib.edgetpu_list_devices.restype = POINTER(EdgetpuDevice)
        self.lib.edgetpu_list_devices.argtypes = [POINTER(c_size_t)]
        self.lib.edgetpu_free_devices.argtypes = [ctypes.c_void_p]

    def version(self):
        return self.lib.edgetpu_version().decode("utf-8", "replace")

    def set_verbosity(self, level):
        self.lib.edgetpu_verbosity(int(level))

    def list_devices(self):
        count = c_size_t(0)
        raw = self.lib.edgetpu_list_devices(ctypes.byref(count))
        devices = []
        for i in range(count.value):
            dev = raw[i]
            devices.append({
                "type": int(dev.type),
                "path": dev.path.decode("utf-8", "replace") if dev.path else None,
            })
        self.lib.edgetpu_free_devices(raw)
        return devices
