import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nuztf",
    version="2.4.2",
    author="Robert Stein, Simeon Reusch, Jannis Necker",
    author_email="rdstein@caltech.edu, simeon.reusch@desy.de, jannis.necker@desy.de",
    description="Package for multi-messenger correlation searches with ZTF",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords="astroparticle physics science multimessenger astronomy ZTF",
    url="https://github.com/desy-multimessenger/nuztf",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8.0,<3.9.0",
    install_requires=[
        "ampel-alerts == 0.7.2",
        "ampel-core == 0.7.4",
        "ampel-interface == 0.7.1",
        "ampel-photometry == 0.7.1",
        "ampel-ztf == 0.7.4.post1",
        "astropy == 5.1",
        "black == 22.6.0",
        "backoff == 1.11.1",
        "coveralls == 3.3.1",
        "geopandas == 0.10.2",
        "gwemopt == 0.0.76",
        "healpy == 1.15.2",
        "ipykernel == 6.15.1",
        "jupyter == 1.0.0",
        "ligo-gracedb == 2.7.6",
        "ligo.skymap == 1.0.0",
        "lxml==4.9.1",
        "matplotlib==3.5.1",
        "mocpy==0.11.0",
        "numpy==1.23.0",
        "pandas == 1.3.5",
        "pre_commit == 2.20.0",
        "python-ligo-lw == 1.8.0",
        "pyvo == 1.2.1",
        "requests == 2.28.1",
        "seaborn == 0.11.2",
        "setuptools == 63.1.0",
        "tqdm == 4.64.0",
        "wget == 3.2",
        "ztfquery == 1.18.4",
    ],
)
