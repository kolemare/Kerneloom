from invoke import task
from pathlib import Path
import shutil


BUILD_DIR = Path("build")
TEST_BUILD_DIR = Path("build_tests")

APP_BINARY = BUILD_DIR / "kerneloom"
TEST_BINARY = TEST_BUILD_DIR / "tests" / "kerneloom_tests"


def cmake_build(ctx, build_dir, cuda=False, rocm=False, debug=False, tests=False, jobs=16):
    if cuda and rocm:
        raise ValueError("choose only one backend: --cuda or --rocm")

    args = [f"-DCMAKE_BUILD_TYPE={'Debug' if debug else 'Release'}"]

    if cuda:
        args.append("-DKL_ENABLE_CUDA=ON")
    elif rocm:
        args.append("-DKL_ENABLE_ROCM=ON")

    if tests:
        args.append("-DKL_BUILD_TESTS=ON")
        args.append("-DKL_BUILD_APP=OFF")

    ctx.run(f"cmake -S . -B {build_dir} {' '.join(args)}", pty=True)
    ctx.run(f"cmake --build {build_dir} -j {jobs}", pty=True)


def remove(path):
    if path.exists():
        shutil.rmtree(path)


@task
def clean(ctx):
    remove(BUILD_DIR)
    remove(TEST_BUILD_DIR)


@task
def build(ctx, cuda=False, rocm=False, debug=False, jobs=16):
    remove(BUILD_DIR)
    cmake_build(ctx, BUILD_DIR, cuda, rocm, debug, False, jobs)


@task(name="build-tests")
def build_tests(ctx, cuda=False, rocm=False, debug=False, jobs=16):
    remove(TEST_BUILD_DIR)
    cmake_build(ctx, TEST_BUILD_DIR, cuda, rocm, debug, True, jobs)


@task
def test(ctx, cuda=False, rocm=False, debug=False, jobs=16):
    build_tests(ctx, cuda, rocm, debug, jobs)
    ctx.run(str(TEST_BINARY), pty=True)


@task(name="builds")
def builds(ctx, debug=False, jobs=16):
    for name, opts in [
        ("CPU", {}),
        ("CUDA", {"cuda": True}),
        ("ROCm", {"rocm": True}),
    ]:
        print(f"\n=== Building {name} ===\n")
        build(ctx, debug=debug, jobs=jobs, **opts)


@task(name="test-builds")
def test_builds(ctx, debug=False, jobs=16):
    for name, opts in [
        ("CPU Tests", {}),
        ("CUDA Tests", {"cuda": True}),
        ("ROCm Tests", {"rocm": True}),
    ]:
        print(f"\n=== Building {name} ===\n")
        build_tests(ctx, debug=debug, jobs=jobs, **opts)


@task
def run(ctx):
    ctx.run(str(APP_BINARY), pty=True)


@task(name="run-tests")
def run_tests(ctx):
    ctx.run(str(TEST_BINARY), pty=True)


@task(name="run-test-binary")
def run_test_binary(ctx):
    ctx.run(str(TEST_BINARY), pty=True)