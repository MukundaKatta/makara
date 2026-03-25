"""Setup configuration for the Makara package."""

from setuptools import setup, find_packages

setup(
    name="makara",
    version="0.1.0",
    description="Web intelligence platform - scrape, analyze, and summarize web content using AI agents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="MukundaKatta",
    url="https://github.com/MukundaKatta/makara",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
