"""Command-line interface for the IREE Coral Edge TPU toolchain.

Subcommands:

  doctor    diagnose the installation (build tools, artifacts, hardware)
  version   print the pinned versions
  devices   enumerate Edge TPU devices
  compile   compile a TFLite model to an Edge TPU executable (edgetpu_compiler)
  run       import + compile a TFLite model through the Coral IREE pipeline
  test      run the toolchain test suites
"""

import argparse
import os
import subprocess
import sys

from . import doctor as doctor_mod
from . import paths
from . import version as version_mod
from .compiler import compile_model
from .runtime import CoralRuntime


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="iree-coral",
        description="IREE Coral Edge TPU toolchain helper")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="Diagnose the installation")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("version", help="Print pinned versions")
    p.set_defaults(func=cmd_version)

    p = sub.add_parser("devices", help="Enumerate Edge TPU devices")
    p.set_defaults(func=cmd_devices)

    p = sub.add_parser("compile",
                       help="Compile a TFLite model with edgetpu_compiler")
    p.add_argument("input")
    p.add_argument("-o", "--out-dir", required=True)
    p.set_defaults(func=cmd_compile)

    p = sub.add_parser(
        "run", help="Import + compile a TFLite model for the Coral IREE runtime")
    p.add_argument("input", help="quantized .tflite model")
    p.add_argument("-o", "--out-dir", required=True,
                   help="directory for the .mlir and .vmfb")
    p.add_argument("--tag", default="model")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("test", help="Run the toolchain test suites")
    p.add_argument("suites", nargs="*", help="suite file names (default: all)")
    p.set_defaults(func=cmd_test)

    args = parser.parse_args(argv)
    return args.func(args)


def cmd_doctor(args):
    return doctor_mod.run(args)


def cmd_version(args):
    print(version_mod.summary())
    for key, val in version_mod.details():
        print(f"  {key:24s} {val}")
    return 0


def cmd_devices(args):
    rt = CoralRuntime()
    devices = rt.list_devices()
    print(f"libedgetpu {rt.version()}")
    print(f"devices: {len(devices)}")
    for dev in devices:
        print(f"  type={dev['type']} path={dev['path']}")
    return 0 if devices else 1


def cmd_compile(args):
    result = compile_model(args.input, args.out_dir)
    print(result["stdout"])
    if result["output_path"]:
        print(f"wrote {result['output_path']}")
        return 0
    return 1


def cmd_run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    mlir = os.path.join(args.out_dir, "model.mlir")
    vmfb = os.path.join(args.out_dir, "model.vmfb")

    if not os.path.exists(paths.IREE_COMPILE):
        print(f"iree-compile not found: {paths.IREE_COMPILE}", file=sys.stderr)
        return 1

    env = paths.compiler_env()

    print(f"importing {args.input} -> {mlir}")
    rc = subprocess.run([sys.executable, paths.IMPORTER, args.input,
                         "-o", mlir, "--tag", args.tag], env=env).returncode
    if rc != 0:
        return rc

    print(f"compiling {mlir} -> {vmfb}")
    return subprocess.run([paths.IREE_COMPILE, "--iree-hal-target-backends=coral",
                           mlir, "-o", vmfb], env=env).returncode


def cmd_test(args):
    script = os.path.join(paths.TOOLCHAIN_ROOT, "scripts", "run_all_tests.sh")
    if not os.path.exists(script):
        print(f"test driver not found: {script}", file=sys.stderr)
        return 1
    return subprocess.run(["bash", script, *args.suites]).returncode


if __name__ == "__main__":
    sys.exit(main())
