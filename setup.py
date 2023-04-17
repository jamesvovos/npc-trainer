from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "0.0.1"
DESCRIPTION = "Creating/training NPC AI models using Pytorch and spaCy"
LONG_DESCRIPTION = "A package that allows you to build your own NPC AI models"

# Setting up
setup(
    name="npc-ai-trainer",
    version=VERSION,
    author="James Vovos",
    author_email="<james.vovos@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    keywords=["python", "ai", "pytorch", "npc", "training", "spaCy"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
