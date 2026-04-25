from setuptools import setup, find_packages

setup(
    name="Bandit",
    version="0.1.0",
    description="Bandit algorithms for research",
    author="Kishan Chakraborty",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    python_requires=">=3.8",
)
