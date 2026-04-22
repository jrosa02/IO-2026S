from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext
from pathlib import Path
import shutil

import numpy as np


class build_ext(_build_ext):
    """Custom build_ext that places compiled .so files in src/native_optimized/_compiled"""

    def build_extension(self, ext):
        """Override to capture and display pragma messages during compilation"""
        # Temporarily capture compiler output to display pragma messages
        import subprocess

        # Get the compiler command that would be used
        sources = self.sources if hasattr(self, 'sources') else ext.sources

        print(f"\n{'='*70}")
        print(f"Compiling {ext.name}...")
        print(f"{'='*70}")

        # Call parent build_extension
        super().build_extension(ext)

    def run(self):
        super().run()

        # After all extensions are built, copy them to _compiled directory
        compiled_dir = Path(__file__).parent / "src" / "native_optimized" / "_compiled"
        compiled_dir.mkdir(exist_ok=True)

        print(f"\n{'='*70}")
        print("Installing compiled extensions to _compiled/")
        print(f"{'='*70}")

        # Copy from build output to _compiled and remove from src/
        src_dir = Path(__file__).parent / "src"
        for root, dirs, files in Path(self.build_lib).walk():
            for file in files:
                if file.endswith('.so'):
                    src = Path(root) / file
                    dest = compiled_dir / file
                    print(f"  → {dest}")
                    shutil.copy2(src, dest)

                    # Remove from src/ if it exists there
                    src_so = src_dir / file
                    if src_so.exists():
                        src_so.unlink()


setup(
    ext_modules=[
        Extension(
            "src.evaluate_opt",
            sources=["src/native_optimized/evaluate_opt.c"],
            include_dirs=[np.get_include()],
            extra_compile_args=[
                "-O3",
                "-march=native",
                "-ffast-math",
                "-funroll-loops",
                "-fomit-frame-pointer",
                "-fno-plt",
                "-std=c11",
            ],
        ),
        Extension(
            "src.crossover_opt",
            sources=["src/native_optimized/crossover_opt.c"],
            include_dirs=[np.get_include()],
            extra_compile_args=[
                "-O3",
                "-march=native",
                "-ffast-math",
                "-funroll-loops",
                "-fomit-frame-pointer",
                "-fno-plt",
                "-std=c11",
            ],
        )
    ],
    cmdclass={"build_ext": build_ext}
)
