from distutils.core import setup
setup(
  name = 'brain4k',
  packages = ['brain4k'],
  version = '0.1',
  description = 'A framework for machine learning pipelines',
  author = 'Robert Kyle',
  author_email = 'rob@homerundata.com',
  url = 'https://github.com/shuggiefisher/brain4k',
  download_url = 'https://github.com/shuggiefisher/brain4k/tarball/0.1',
  keywords = ['machine', 'learning', 'pipeline', 'deep', 'neural', 'network'],
  classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.2',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
  ],
  install_requires = ['numpy', 'pandas', 'h5py'],
  entry_points = {
    'console_scripts': ['brain4k = brain4k.brain4k:run'],
    },
)