import os


def env_configure():
    os.environ["CUDA_VERSION"] = "12.1"
    os.environ["CUDA_HOME"] = f'/mnt/public/lib/cuda/cuda-{os.environ["CUDA_VERSION"]}'
    os.environ["PATH"] = f'{os.environ["CUDA_HOME"]}/bin:{os.environ["PATH"]}'
    os.environ["LD_LIBRARY_PATH"] = (
        f'{os.environ["CUDA_HOME"]}/lib64:{os.environ["LD_LIBRARY_PATH"]}'
    )


