"""Setup configurations."""
from setuptools import setup, find_packages

setup(
    name='creative_ai',
    version='0.1.0',
    packages=['creative_ai'],
    include_package_data=True,
    install_requires=[
        'tqdm',
        'wheel',
        'pysynth @ https://github.com/eecs183/183pysynth/archive/master.zip',
        'spacy',
        'tweepy'
    ]
)
