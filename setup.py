from setuptools import setup, find_packages

setup(name='DQNCloud',
  version='0.0.1',
  install_requires=['pymongo', 'torch', 'numpy', 'flask', 'sklearn', 'matplotlib'],
  description='A cloud service for training and optimizing a DQN',
  author = 'Daniel Fredriksen',
  author_email = 'dfredriksen@cyint.technology',
  url='https://github.com/dfredriksen/DQNCloud',
  packages=find_packages(where='DQNCloud'),
  include_package_data=True
)
