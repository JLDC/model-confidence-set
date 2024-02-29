from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name="model-confidence-set",
    version="0.1.0",
    license="MIT",
    description="model-confidence-set provides a Python implementation of the Model Confidence Set (MCS) procedure (Hansen, Lunde, and Nason, 2011), a statistical method for comparing and selecting models based on their performance.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jonathan Chassot",
    author_email="jonathan.chassot@unisg.ch",
    url="https://github.com/JLDC/model-confidence-set",
    keywords=["model confidence set", "model evaluation", 
              "statistical model comparison", "model performance analysis",
              "model selection", "predictive accuracy", "econometrics", 
              "financial econometrics"],
    install_requires=[
        "numba>=0.59.0",
        "numpy>=1.26.4",
        "pandas>=2.2.1",
        "tqdm>=4.66.2"
    ]
)