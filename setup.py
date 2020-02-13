import setuptools, sys, os
import subprocess
from SPCA import __version__, name

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ["numpy", "scipy", "astropy", "matplotlib", "emcee",
                    "batman-package", "corner", "photutils", "pandas", "pyyaml", "threadpoolctl"]


response = input('Would you like to install the optional package mc3 for nicer rednoise plots? (y/n)')
if response.lower()=='y':
    installMC3 = True
else:
    installMC3 = False

response = input('Would you like to install the optional package george to perform GP decorrelations? (y/n)')
if response.lower()=='y':
    install_requires.append("george")

setuptools.setup(
    name = name,
    version = __version__,
    author = "Lisa Dang & Taylor James Bell",
    author_email = "lisa.dang@physics.mcgill.ca & taylor.bell@mail.mcgill.ca",
    description = "An open-source, modular, and automated pipeline for Spitzer Phase Curve Analyses.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/lisadang27/SPCA",
    license="GPL3",
    package_data={"": ["LICENSE"]},
    packages = setuptools.find_packages(),
    classifiers = (
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GPL3 License",
        "Operating System :: OS Independent",
    ),
    include_package_data = True,
    zip_safe = True,
    install_requires = [
        "numpy", "scipy", "astropy", "matplotlib", "emcee", "batman-package", "corner", "photutils", "pandas", "pyyaml", "threadpoolctl"]
)

if installMC3:
    # Install the MC3 package
    print('Installing MC3.')
    with subprocess.Popen(['git', 'clone', 'https://github.com/pcubillos/MCcubed'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as out:
        stdout,stderr = out.communicate()
        print('Output:', stdout.decode())
        print('Errors:', stderr.decode())

    os.system('cd MCcubed')

    with subprocess.Popen(['make'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as out:
        stdout,stderr = out.communicate()
        print('Output:', stdout.decode())
        print('Errors:', stderr.decode())

    with subprocess.Popen(['pip', 'install', '.'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as out:
        stdout,stderr = out.communicate()
        print('Output:', stdout.decode())
        print('Errors:', stderr.decode())

    os.system('cd ../')
