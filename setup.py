from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


SOURCE = str(Path("native") / "native_standard_successors.cpp")


def extension(module: str) -> Pybind11Extension:
    short_name = module.rsplit(".", 1)[-1]
    return Pybind11Extension(
        module,
        [SOURCE],
        define_macros=[("DUAL_GS_MODULE", short_name)],
        cxx_std=17,
    )


setup(
    ext_modules=[
        extension("ac_final_paper.solvers.native_compact_dual_gs"),
        extension("ac_final_paper.solvers.native_compact_triple_gs"),
    ],
    cmdclass={"build_ext": build_ext},
)
