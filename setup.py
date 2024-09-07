from setuptools import setup, find_packages

setup(

    name='SS_SPAM',

    version='0.1.0',

    author='MapleleavesCX / Rex / sdushushu / Taorich',

    author_email='None',

    description='A privacy preserving spam filter based on SecretFlow',

    long_description=open('README.md').read(),

    long_description_content_type='text/markdown',

    url='https://github.com/MapleleavesCX/SS_SPAM/',

    packages=find_packages(),

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: None',
        'Operating System :: POSIX :: Linux',
    ],

    python_requires='>=3.10',

    install_requires=[
        'secretflow',
        'scikit-learn',
        'nltk',
        'pandas'
    ],

)
