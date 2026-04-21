from setuptools import Extension, setup

import numpy as np

setup(
    ext_modules=[
        Extension(
            "src.optimized",
            sources=["src/native_optimized/optimized.c"],
            include_dirs=[np.get_include()],
            extra_compile_args=[
                "-O3",
                "-march=znver4",               # Zen 4: enables AVX-512F/DQ/BW/VL + BMI2 + FMA
                "-mtune=znver4",               # instruction scheduling for Zen 4 µarch
                "-mprefer-vector-width=512",   # Zen 4: 512-bit ops == 256-bit throughput
                "-mavx512f", "-mavx512dq",     # native mullo_epi64, 8-wide int64
                "-ffast-math",
                "-funroll-loops",
                "-fprefetch-loop-arrays",
                "-fomit-frame-pointer",
                "-fno-plt",
                "-fvisibility=hidden",
                "-fvect-cost-model=unlimited", # don't suppress SIMD on cost grounds
                "-std=c11",
            ],
        )
    ]
)
