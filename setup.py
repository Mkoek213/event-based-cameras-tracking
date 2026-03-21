from setuptools import setup, find_packages

setup(
    name="event-based-cameras-tracking",
    version="0.1.0",
    description="Object tracking using event-based cameras",
    author="Mkoek213",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.7.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
    ],
)
