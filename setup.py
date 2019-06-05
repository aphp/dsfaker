from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='dsfaker',
      version='0.2.3',
      description='Data Science Faker',
      long_description=readme(),
      url='http://github.com/dubrzr/dsfaker',
      author='dubrzr',
      author_email='nota@valid.addr',
      license='MIT',
      packages=['dsfaker',
                'dsfaker.generators',
                'dsfaker.noise'],
      install_requires=[
            'numpy>=1.12.0',
      ],
      zip_safe=False)
