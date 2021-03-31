from setuptools import find_packages, setup

setup(
    name="lsnms",
    version="0.1.1",
    description="Large Scale Non Maximum Suppression",
    author="Rémy Dubois",
    install_requires=["numpy==1.19.5", "numba==0.53.1"],
    url="https://github.com/remydubois/lsnms",
    keywords=[
        "NMS",
        "Non Maximum Suppression",
        "Histology image",
        "Satellite images",
        "Object detection",
        "Large scale processing",
    ],
    classifiers=["Programming Language :: Python :: 3", "Topic :: Scientific/Engineering"],
    packages=find_packages(exclude=("tests")),
)
