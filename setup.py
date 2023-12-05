from setuptools import setup, find_packages  # type: ignore

setup(
    name="scaffold",
    version="0.0.1",
    author="Aaron Mishkin",
    author_email="amishkin@cs.stanford.edu",
    description="Experiments for ICML 2022 paper.",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    package_data={"scaffold": ["py.typed"]},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "torchaudio",
        "cvxpy",
        "scikit-learn",
        "scipy",
        "tqdm",
        "opt-einsum",
        "matplotlib",
        "xgboost",
        "linalg_backends @ git+https://git@github.com/aaronpmishkin/lab#egg=lab",
        "experiment_utils @ git+https://git@github.com/aaronpmishkin/experiment_utils#egg=experiment_utils",
        "pyscnn @ git+https://git@github.com/pilancilab/scnn@arbitrary_gates#egg=scnn",
    ],
)
