from setuptools import setup, find_packages  # type: ignore

setup(
    name="scaffold",
    version="0.0.1",
    author="Aaron Mishkin",
    author_email="amishkin@cs.stanford.edu",
    description="Experiments for convex_nn submission.",
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
        "numpy==1.21.3",
        "torch",
        "torchvision",
        "torchaudio",
        "cvxpy==1.1.15",
        "scikit-learn",
        "scipy",
        "tqdm",
        "opt-einsum",
        "matplotlib",
        # "lab @ git+https://git@github.com/aaronpmishkin/lab@main#egg=lab",
        # "experiment_utils @ git+https://git@github.com/aaronpmishkin/experiment_utils@main#egg=experiment_utils",
        # "convex_nn @ git+https://git@github.com/aaronpmishkin/convex_nn@main#egg=convex_nn",
    ],
)
