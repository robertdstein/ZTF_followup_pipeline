import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nuztf",
    version="2.4.1",
    author="Robert Stein, Simeon Reusch, Jannis Necker",
    author_email="robert.stein@desy.de, simeon.reusch@desy.de, jannis.necker@desy.de",
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
        "ampel-ztf == 0.7.4",
        "astropy == 4.3.1",
        "backoff == 1.11.1",
        "coveralls == 3.3.1",
        "fitsio == 1.1.5",
        "geopandas == 0.10.2",
        "gwemopt == 0.0.73",
        "healpy == 1.15.0",
        "ipykernel == 6.6.0",
        "jupyter == 1.0.0",
        "ligo-gracedb == 2.7.6",
        "ligo.skymap == 0.5.3",
        "lxml==4.6.5",
        "matplotlib==3.5.0",
        "numpy==1.21.4",
        "pandas == 1.3.5",
        "python-ligo-lw == 1.7.1",
        "requests == 2.26.0",
        "seaborn == 0.11.2",
        "setuptools == 59.5.0",
        "tqdm == 4.62.3",
        "wget == 3.2",
        "ztfquery == 1.18.0",
    ],
)
