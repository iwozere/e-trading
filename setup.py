from setuptools import find_packages, setup

setup(
    name="crypto-trading",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.2.0",
        "pandas>=2.2.0",
        "scikit-optimize>=0.10.0",
        "scikit-learn>=1.4.0",
        "scipy>=1.12.0",
        "TA-Lib>=0.4.0",
        "python-dateutil>=2.8.0",
        "pytz>=2024.1",
    ],
)
