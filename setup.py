import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nuztf",
    version="2.1.0",
    author="Robert Stein",
    author_email="robert.stein@desy.de",
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
    python_requires='>=3.8.0,<3.9.0',
    install_requires=[
        "ampel-alerts == 0.7.2",
        "ampel-core == 0.7.4",
        "ampel-interface == 0.7.1",
        "ampel-photometry == 0.7.1",
        "ampel-ztf == 0.7.3",
        "astrobject == 0.8.6",
        "astropy == 4.2.1",
        "astropy_healpix == 0.6",
        "backoff == 1.11.1",
        "bs4 == 0.0.1",
        "catsHTM == 0.1.32",
        "coveralls == 3.2.0",
        "datetime == 4.3",
        "extcats == 2.4.1",
        "fastavro == 1.4.4",
        "fitsio == 1.1.5",
        "gwemopt == 0.0.73",
        "healpy == 1.15.0",
        "ipykernel == 6.4.1",
        "jupyter == 1.0.0",
        "ligo-gracedb == 2.7.6",
        "lxml==4.6.3",
        "matplotlib==3.4.3",
        "numpy==1.21.2",
        "pandas == 1.3.3",
        "psycopg2-binary == 2.9.1",
        "pydantic == 1.4",
        "pymongo == 3.12.0",
        "pysedm == 0.27.4",
        "pyvo == 1.1",
        "ratelimit == 2.2.1",
        "requests == 2.26.0",
        "scipy == 1.7.1",
        "setuptools == 58.0.2",
        "shapely == 1.7.1",
        "sklearn == 0.0",
        "slackclient == 2.9.3",
        "sqlalchemy == 1.4.23",
        "tqdm == 4.62.2",
        "wget == 3.2",
        "zerorpc == 0.6.3",
        "ztf-plan-obs == 0.33",
        "ztfquery == 1.15.9"
    ]
)

