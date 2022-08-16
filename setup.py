from setuptools import setup, find_packages

setup(
    name='multiagent_puzzles',
    version='0.0.1',
    description='Multi-Agent Puzzle Environment',
    url='https://github.com/khajash/multiagent-env',
    author='Kate Hajash',
    author_email='kshajash@gmail.com.com',
    packages=find_packages(),
    install_requires=[
        'gym==0.21',
        'numpy', 
        'box2d-py',
        'pyglet'
    ]
)