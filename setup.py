import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mcroute", # Replace with your own username
    version="1.0.0",
    author="Willem Klumpenhouwer",
    author_email="willem@klumpentown.com",
    description="Markov chain modelling of transportation networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wklumpen/mcroute",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
    ],
    python_requires='>=3.6',
    install_requires=[
        'decorator',
        'networkx',
        'numpy',
        'scipy',
    ]
)