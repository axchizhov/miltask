from setuptools import setup, find_packages, find_namespace_packages

setup(
    name='miltask',
    version='0.1',
    packages=find_packages(
        exclude=['tests', 'data', 'outputs']
    )
)
