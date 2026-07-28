from invoke import task
from pathlib import Path
import shutil
import shlex


BUILD_DIR = Path("build")

APP_BINARY = BUILD_DIR / "kerneloom"
TEST_BINARY = BUILD_DIR / "tests" / "kerneloom_tests"


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


def test_filter(name):
    if not name:
        return None

    if "." not in name:
        return f"{name}.*"

    return name


def disabled_filter(name):
    if not name:
        return "DISABLED_*:*DISABLED_*"

    if "." not in name:
        return f"DISABLED_{name}.*:{name}.DISABLED_*"

    suite, test = name.split(".", 1)

    return (
        f"DISABLED_{suite}.{test}:"
        f"{suite}.DISABLED_{test}:"
        f"DISABLED_{suite}.DISABLED_{test}"
    )


def gtest_args(filter=None, repeat=0, disabled=False):
    args = []

    selected_filter = disabled_filter(filter) if disabled else test_filter(filter)

    if selected_filter:
        args.append(f"--gtest_filter={shlex.quote(selected_filter)}")

    if disabled:
        args.append("--gtest_also_run_disabled_tests")

    if repeat:
        args.append(f"--gtest_repeat={repeat}")

    return " ".join(args)


@task
def clean(ctx):
    remove(BUILD_DIR)


@task
def build(ctx, cuda=False, rocm=False, cuda_tests=False, rocm_tests=False, debug=False, jobs=16):
    remove(BUILD_DIR)

    if cuda_tests or rocm_tests:
        cmake_build(ctx, BUILD_DIR, cuda_tests, rocm_tests, debug, True, jobs)
    else:
        cmake_build(ctx, BUILD_DIR, cuda, rocm, debug, False, jobs)


@task
def run(ctx):
    ctx.run(str(APP_BINARY), pty=True)


@task(name="test")
def test(ctx, filter=None, repeat=0):
    ctx.run(f"{TEST_BINARY} {gtest_args(filter, repeat)}", pty=True)


@task
def spin(ctx, filter=None, repeat=0):
    ctx.run(f"{TEST_BINARY} {gtest_args(filter, repeat, disabled=True)}", pty=True)