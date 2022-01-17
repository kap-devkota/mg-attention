from setuptools import setup, find_packages

setup(
    name='mgattention',
    version='1',
    description='Multi-graph attention',
    author='Kapil Devkota',
    author_email='kapil.devkota@tufts.edu',
    url='https://github.com/kap-devkota/mg-attention.git',
    packages=find_packages(exclude=('tests', 'docs', 'results', 'data')),
    package_dir={'mgattention':'mgattention'}
)
