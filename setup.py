from setuptools import setup, find_packages

setup(
    name="vlhf",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[line.strip() for line in open("requirements.txt")],
)
