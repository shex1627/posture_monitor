from setuptools import setup, find_packages


# with open('README.rst') as f:
#     readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='posture_monitor',
    version='0.1.0',
    description='no description',
    long_description="no description",
    author='shicheng huang',
    author_email='shicheng1627@gmail.com',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)