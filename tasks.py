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
def build(ctx, cuda=False, rocm=False, jobs=8):
    if cuda and rocm:
        raise ValueError("choose only one backend: --cuda or --rocm")

    cmake_args = []

    if cuda:
        cmake_args.append("-DKL_ENABLE_CUDA=ON")
    elif rocm:
        cmake_args.append("-DKL_ENABLE_ROCM=ON")

    ctx.run(f"cmake -S . -B {BUILD_DIR} {' '.join(cmake_args)}", pty=True)
    ctx.run(f"cmake --build {BUILD_DIR} -j {jobs}", pty=True)


@task
def run(ctx):
    ctx.run(str(BINARY), pty=True)