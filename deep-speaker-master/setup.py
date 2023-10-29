import os
import platform

from setuptools import setup

tensorflow = 'tensorflow'
if platform.system() == 'Darwin' and platform.processor() == 'arm':
    tensorflow = 'tensorflow-macos'
    # https://github.com/grpc/grpc/issues/25082
    os.environ['GRPC_PYTHON_BUILD_SYSTEM_OPENSSL'] = '1'
    os.environ['GRPC_PYTHON_BUILD_SYSTEM_ZLIB'] = '1'

install_requires = [
    'numpy',
    tensorflow,
    'natsort',
    'librosa',
    'dill',
    'python_speech_features',
    'tqdm',
    'click',
    'pandas',
    'matplotlib',
    'keras',
    'numba'
]

setup(
    name='deep-speaker',
    version='1.0',
    description='Deep Speaker',
    author='Philippe Remy',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['deep_speaker'],
    install_requires=install_requires
)
