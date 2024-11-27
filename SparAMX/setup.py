import os
import glob
import shutil
import torch
from setuptools import setup, find_packages, Command
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME
from pathlib import Path

# Get PyTorch library path
TORCH_LIB_PATH = str(Path(torch.__file__).parent / 'lib')

# Add torch lib path to environment
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] = f"{TORCH_LIB_PATH}:{os.environ['LD_LIBRARY_PATH']}"
else:
    os.environ['LD_LIBRARY_PATH'] = TORCH_LIB_PATH

class CustomCleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        patterns_to_remove = [
            'sparamx.egg-info',
            'sparamx/*.so',
            'sparamx/*.pyd',
        ]

        build_dirs = ['./build', './dist']
        for dir_path in build_dirs:
            if os.path.exists(dir_path):
                print(f'Removing directory: {dir_path}')
                shutil.rmtree(dir_path)

        for pattern in patterns_to_remove:
            for item in glob.glob(pattern):
                if os.path.isdir(item):
                    print(f'Removing directory: {item}')
                    shutil.rmtree(item)
                elif os.path.isfile(item):
                    print(f'Removing file: {item}')
                    os.remove(item)

        for root, dirs, files in os.walk('./sparamx'):
            if '__pycache__' in dirs:
                cache_dir = os.path.join(root, '__pycache__')
                print(f'Removing directory: {cache_dir}')
                shutil.rmtree(cache_dir)

# Common compiler and linker arguments
extra_compile_args = [
    '-mamx-tile', 
    '-mamx-int8', 
    '-mamx-bf16', 
    '-fopenmp', 
    '-O3', 
    '-DNDEBUG', 
    '-march=sapphirerapids', 
    '-mavx512f', 
    '-mavx512dq'
]

# Add PyTorch include paths
include_dirs = [
    os.path.join(torch.utils.cpp_extension.include_paths()[0], 'torch', 'csrc', 'api', 'include'),
    os.path.join(torch.utils.cpp_extension.include_paths()[0], 'torch', 'lib'),
]

# Define extensions
extension_specs = [
    ("sparse_linear", "csrc/sparse_linear.cpp"),
    ("avx_sparse_linear", "csrc/avx_sparse_linear.cpp"),
    ("quantized_sparse_linear", "csrc/quantized_sparse_linear.cpp"),
    ("quantized_dense_linear", "csrc/quantized_dense_linear.cpp"),
    ("dense_linear", "csrc/dense_linear.cpp"),
]

extensions = []
for name, source in extension_specs:
    source_path = os.path.abspath(source)
    print(f"Setting up extension {name} from source {source_path}")
    
    if not os.path.exists(source_path):
        print(f"WARNING: Source file not found: {source_path}")
        continue
        
    ext = CppExtension(
        name=f"sparamx.{name}",
        sources=[source_path],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=['-lgomp', f'-Wl,-rpath,{TORCH_LIB_PATH}']
    )
    extensions.append(ext)

setup(
    name="sparamx",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass={
        'build_ext': BuildExtension,
        'clean': CustomCleanCommand,
    }
)