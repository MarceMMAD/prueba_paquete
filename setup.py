#from distutils.core import setup
from setuptools import setup

setup(
  name = 'prueba_paquete',
  packages = ['prueba_paquete'], # this must be the same as the name above
  version = '0.1.3',
  description = 'NLP and ML modules for processing civic input',
  author = 'Marcelo Alcaraz',
  author_email = 'marce.mmad@gmail.com',
  url = 'https://github.com/MarceMMAD/prueba_paquete', # use the URL to the github repo
  download_url = 'https://github.com/MarceMMAD/prueba_paquete/0.1.3.tar.gz', # I'll explain this in a second
  keywords = ['Natural Language Processing', 'Machine Learnign', 'Civic input'], # arbitrary keywords
  classifiers = [],
  install_requires=['nltk', 'sklearn', 'bs4', 'googletrans'],
  python_requires='>=3',
  package_data={
    'mlsenticon': ['lexicon_lib/MLsenticon.es.xml'],
    'new_words:': ['lexicon_lib/new_neg_words.txt', 'lexicon_lib/new_pos_words'],
    'spanish_lemmas': ["lexicon_lib/lemmatization-es.txt"]
  },
)