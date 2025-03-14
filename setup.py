from setuptools import setup, find_packages

setup(
    name='andylib',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python'
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'flake8>=4.0.0',
            'black>=22.0.0',
            'build',
            'twine',
        ],
    },
    author='Andreas Netsch',
    author_email='your.email@example.com',
    description='A library for image processing functions.',
    url='https://github.com/yourusername/andylib',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
) 