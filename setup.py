import os
import io
from setuptools import setup, find_packages


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()


setup(
    name="moinn",
    version="0.0.1",
    author="Jonas Lederer, Michael Gastegger, Kristof T. Schütt, "
           "Michael Kampffmeyer, Klaus-Robert Müller, Oliver T. Unke",
    email="jonas.lederer@tu-berlin.de",
    url="https://github.com/jnsLs/MoINN",
    packages=find_packages("src"),
    scripts=[
        "src/scripts/moinn_train.py",
    ],
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.1",
        "numpy",
        "ase>=3.18",
        "h5py",
        "tensorboardX",
        "tqdm",
        "pyyaml",
        "schnetpack@git+https://github.com/atomistic-machine-learning/schnetpack/tree/schnetpack1.0.git",
    ],
    extras_require={},
    license="MIT",
    description="MoINN - Automatic Identification of Chemical Moieties",
    long_description="""
        MoINN aims to automatically identify chemical moieties (molecular building blocks) from such machine learned
        representations, enabling a variety of applications beyond property prediction, which otherwise rely on
        expert knowledge.
    """,
)