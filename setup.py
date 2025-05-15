import glob
import os
import subprocess
import sys
import time

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from pathlib import Path

PACKAGE_NAME = "torch_pim"
BUILD_DEPS = True
VERSION = "0.1"

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
THIRD_PARTY_PATH = os.path.join(BASE_DIR, "third_party")

PIMBLAS_BUILD_DIR = os.path.join(THIRD_PARTY_PATH, "libpimblas/build")
PIMBLAS_INSTALL_DIR = os.path.join(THIRD_PARTY_PATH, "libpimblas/install")
PIMBLAS_LIB = os.path.join(PIMBLAS_INSTALL_DIR, "lib")

UPMEMSDK_DIR = os.path.join(THIRD_PARTY_PATH, "upmemsdk")


def get_submodule_folders():
    git_modules_path = os.path.join(BASE_DIR, ".gitmodules")
    with open(git_modules_path) as f:
        return [
            os.path.join(BASE_DIR, line.split("=", 1)[1].strip())
            for line in f.readlines()
            if line.strip().startswith("path")
        ]

def check_submodules():
    def not_exists_or_empty(folder):
        return not os.path.exists(folder) or (
            os.path.isdir(folder) and len(os.listdir(folder)) == 0
        )

    folders = get_submodule_folders()
    # If none of the submodule folders exists, try to initialize them
    if all(not_exists_or_empty(folder) for folder in folders):
        try:
            print(" --- Trying to initialize submodules")
            start = time.time()
            subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=BASE_DIR)  # Compliant
            end = time.time()
            print(f" --- Submodule initialization took {end - start:.2f} sec")
        except Exception:
            print(" --- Submodule initalization failed")
            print("Please run:\n\tgit submodule init && git submodule update")
            sys.exit(1)


def get_csrc():
    csrc_dir = os.path.join(BASE_DIR, "src", PACKAGE_NAME, "csrc")
    return sorted([str(p) for p in Path(csrc_dir).rglob("*.cpp")])


def setup_upmem_env(script_path, default_backend=None):
    script_dir = os.path.dirname(script_path)
    upmem_home = script_dir
    os.environ['UPMEM_HOME'] = upmem_home
    print(f"Setting UPMEM_HOME to {upmem_home} and updating PATH/LD_LIBRARY_PATH/PYTHONPATH")
    ld_lib_path = os.path.join(upmem_home, "lib")
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = f"{ld_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ['LD_LIBRARY_PATH'] = ld_lib_path
    bin_path = os.path.join(upmem_home, "bin")
    os.environ['PATH'] = f"{bin_path}:{os.environ.get('PATH', '')}"
    python_relative_path = "local/lib/python3.10/dist-packages"
    python_path = os.path.join(upmem_home, python_relative_path)
    if 'PYTHONPATH' in os.environ:
        os.environ['PYTHONPATH'] = f"{python_path}:{os.environ['PYTHONPATH']}"
    else:
        os.environ['PYTHONPATH'] = python_path
    if default_backend:
        os.environ['UPMEM_PROFILE_BASE'] = f"backend={default_backend}"
        print(f"Setting default backend to {default_backend} in UPMEM_PROFILE_BASE")

def build_upmemsdk():
    """Install UPMEM SDK via pip and configure environment"""
    print("Installing UPMEM SDK...")
    subprocess.run([sys.executable, "-m", "pip", "install", UPMEMSDK_DIR], check=True)
    result = subprocess.run([sys.executable, "-m", "upmemsdk"], 
                           capture_output=True, text=True, check=True)
    env_script = result.stdout.strip()
    if env_script:
        print(f"UPMEM SDK environment script: {env_script}")
    else:
        raise RuntimeError("UPMEM SDK installation failed - no env script returned")
    setup_upmem_env(env_script)


def build_pimblas():
    try:
        os.makedirs(PIMBLAS_BUILD_DIR, exist_ok=True)
        subprocess.check_call([
            "cmake",
            f"-DCMAKE_INSTALL_PREFIX={os.path.abspath(PIMBLAS_INSTALL_DIR)}",
            ".."
        ], cwd=PIMBLAS_BUILD_DIR)
        subprocess.check_call(["make", "-j"], cwd=PIMBLAS_BUILD_DIR)
        subprocess.check_call(["make", "install"], cwd=PIMBLAS_BUILD_DIR)
        if not os.path.exists(os.path.join(PIMBLAS_LIB, "libpimblas.so")):
            raise RuntimeError("pimblas build failed - library not found")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to build pimblas: {e}")


def main():
    if BUILD_DEPS:
        check_submodules()
        build_upmemsdk()
        build_pimblas() # TODO: make it so that we include pimblas in package and use it from there

    include_directories = [
        BASE_DIR,
        os.path.join(BASE_DIR, "third_party/libpimblas/install/include")
    ]

    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        packages=find_packages(where="src"),
        package_dir = {"": "src"},
        ext_modules=[
            CppExtension(
                name=f'{PACKAGE_NAME}._C',
                sources=get_csrc(),
                include_dirs=include_directories,
                libraries=["pimblas"],
                extra_compile_args=['-std=c++17', '-lstdc++'],
                runtime_library_dirs=[PIMBLAS_LIB],
                extra_link_args=['-L' + PIMBLAS_LIB],
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
        install_requires=[
            'torch',
            'numpy'
        ],
    )


if __name__ == "__main__":
    main()