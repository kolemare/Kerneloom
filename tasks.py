from invoke import task
from pathlib import Path
import shutil


BUILD_DIR = Path("build")
TEST_BUILD_DIR = Path("build_tests")

APP_BINARY = BUILD_DIR / "kerneloom"
TEST_BINARY = TEST_BUILD_DIR / "tests" / "kerneloom_tests"


def remove(path):
    if path.exists():
        shutil.rmtree(path)


def cmake_build(ctx, build_dir, cuda=False, rocm=False, debug=False, tests=False, jobs=16):
    if cuda and rocm:
        raise ValueError("choose only one backend: --cuda or --rocm")

    args = [f"-DCMAKE_BUILD_TYPE={'Debug' if debug else 'Release'}"]

    if cuda:
        args.append("-DKL_ENABLE_CUDA=ON")
    elif rocm:
        args.append("-DKL_ENABLE_ROCM=ON")

    if tests:
        args += ["-DKL_BUILD_TESTS=ON", "-DKL_BUILD_APP=OFF"]

    ctx.run(f"cmake -S . -B {build_dir} {' '.join(args)}", pty=True)
    ctx.run(f"cmake --build {build_dir} -j {jobs}", pty=True)


@task
def clean(ctx):
    remove(BUILD_DIR)
    remove(TEST_BUILD_DIR)


@task
def build(ctx, cuda=False, rocm=False, cuda_tests=False, rocm_tests=False, debug=False, jobs=16):
    if cuda_tests or rocm_tests:
        remove(TEST_BUILD_DIR)
        cmake_build(ctx, TEST_BUILD_DIR, cuda_tests, rocm_tests, debug, True, jobs)
    else:
        remove(BUILD_DIR)
        cmake_build(ctx, BUILD_DIR, cuda, rocm, debug, False, jobs)


@task
def run(ctx):
    ctx.run(str(APP_BINARY), pty=True)


@task(name="test")
def test(ctx):
    ctx.run(str(TEST_BINARY), pty=True)