from setuptools import setup
import os

current_folder = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_folder, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pclines',
    version='1.0.1',
    description='PCLines transform for python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/RomanJuranek/pclines-python',
    author='Roman Juranek',
    author_email='ijuranek@fit.vutbr.cz',
    license='BSD3',
    keywords='pclines, hough transform, line detection',
    packages=["pclines"],
    install_requires=['numpy','numba','scikit-image']
    python_requires='>=3.6',
    project_urls={
        "Bug reports": 'https://github.com/RomanJuranek/pclines-python/issues',
    },
)