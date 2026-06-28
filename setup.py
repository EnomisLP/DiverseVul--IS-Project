from setuptools import setup, find_packages

setup(
    name="vuln-detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "pyreft>=0.0.4",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
    ],
    python_requires=">=3.10",
)