from distutils.core import setup

setup(
  name = 'prueba_paquete',
  packages = ['prueba_paquete'], # this must be the same as the name above
  version = '0.1',
  description = 'A random test lib',
  author = 'Marcelo Alcaraz',
  author_email = 'marce.mmad@gmail.com',
  url = 'https://github.com/MarceMMAD/prueba_paquete', # use the URL to the github repo
  download_url = 'https://github.com/MarceMMAD/prueba_paquete/0.1.tar.gz', # I'll explain this in a second
  keywords = ['testing', 'example'], # arbitrary keywords
  classifiers = [],
)