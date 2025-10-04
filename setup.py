from setuptools import setup

setup(
    name='bootsted_dca',
    packages=['bootsted_dca'],
    include_package_data=True,
    version='0.0.1',
    install_requires=[
        'numpy',
        'matplotlib'
    ],
    author='Bootsted',
    author_email='8Y7yq@example.com',
    description='Bootsted DCA',
    license='MIT',
    url='https://github.com/Bootsted/bootsted_dca',
    python_requires='>=3.6'
)