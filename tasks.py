from invoke import task
from pathlib import Path
import shutil


BUILD_DIR = Path("build")
BINARY = BUILD_DIR / "kerneloom"


@task
def clean(ctx):
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)


@task
def build(ctx, cuda=False, rocm=False, debug=False, jobs=16):
    clean(ctx)

    if cuda and rocm:
        raise ValueError("choose only one backend: --cuda or --rocm")

    build_type = "Debug" if debug else "Release"

    cmake_args = [
        f"-DCMAKE_BUILD_TYPE={build_type}"
    ]

    if cuda:
        cmake_args.append("-DKL_ENABLE_CUDA=ON")
    elif rocm:
        cmake_args.append("-DKL_ENABLE_ROCM=ON")

    ctx.run(f"cmake -S . -B {BUILD_DIR} {' '.join(cmake_args)}", pty=True)
    ctx.run(f"cmake --build {BUILD_DIR} -j {jobs}", pty=True)


@task(name="builds")
def builds(ctx, debug=False, jobs=16):
    try:
        print("\n=== Building CPU ===\n")
        build(ctx, debug=debug, jobs=jobs)

        print("\n=== Building CUDA ===\n")
        build(ctx, cuda=True, debug=debug, jobs=jobs)

        print("\n=== Building ROCm ===\n")
        build(ctx, rocm=True, debug=debug, jobs=jobs)

    finally:
        clean(ctx)


@task
def run(ctx):
    ctx.run(str(BINARY), pty=True)