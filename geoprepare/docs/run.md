### Installing geoprepapre locally
1. Go to directory containing `setup.py`
2. Run `python setup.py install` to install the package locally

### Publish package to PyPI
The steps below follow the instructions [here](https://www.youtube.com/watch?v=7FcX9uWDuIQ)
1. Go to directory containing `setup.py`
2. [OPTIONAL] Update version number in `setup.py`
3. Run `python setup.py sdist` to generate tar to upload to PyPI. This tar file will be in the `dist` folder.
The file name will contain the version number you updated in step 2 e.g. `geoprepare-0.1.3.tar.gz` where `0.1.3` is the version number.
4. Run `twine upload dist/<NAME_OF_TAR_FILE>` to upload the tar file to PyPI.
5. Running step 4 will require the PyPI username and password. If you don't have an account, create one at https://pypi.org/account/register/.
6. Once the upload is complete, go to https://pypi.org/project/geoprepare/ to check that the new version is available.

### Running geoprepare
Whether the package is installed locally or via PyPI, refer to instructions in [README.md](..%2F..%2FREADME.md) for running it.
