from setuptools import setup, find_packages

setup(
    name="soundml",
    version="1.0.0",
    packages=find_packages(),   # finds backend/, backend/api/, backend/ml/, etc.
    install_requires=[
        "scikit-learn",
        "numpy",
        "pandas",
        "joblib",
    ],
)
