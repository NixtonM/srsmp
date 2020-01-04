from setuptools import setup


if __name__ == '__main__':
    setup(name='srsmp',
          version='0.0.1a1',
          description='',
          url='',
          author='Nicholas Meyer, Evelyn Weiss',
          author_email='meyernic@ethz.ch, eweiss@ethz.ch',
          license='GNU General Public License v3.0',
          packages=['srsmp'],
          #scripts=['scripts/01_spectra_sample_generator'],
          install_requires=[
              'cffi',
              'nicelib',
              'numpy',
              'pyvisa',
              #'instrumental-lib',
              'tensorflow==2.0.0',
              'keras==2.3.1',
              'scikit-learn',
              'matplotlib',
              ],
          zip_safe=False,
          )

    