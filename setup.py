from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="NpcTrainerAI",
    version="0.0.2",
    author="James Vovos",
    author_email="<james.vovos@gmail.com>",
    description="Creating/training NPC AI models using Pytorch and spaCy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jamesvovos/npc-trainer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    install_requires=[
        "spacy",
        "requests",
        "torch",
        "pandas",
    ],
    keywords=["python", "ai", "pytorch", "npc", "training", "spaCy"],
)
