from setuptools import setup, find_packages

setup(

    name='your-library-name',

    version='0.1.0',

    author='MapleleavesCX / Rex / sdushushu / ',

    author_email='None',

    description='A short description of your library',

    long_description=open('README.md').read(),

    long_description_content_type='text/markdown',

    url='https://github.com/yourusername/your-library-name',

    packages=find_packages(),

    classifiers=[

        'Programming Language :: Python :: 3',

        'License :: OSI Approved :: MIT License',

        'Operating System :: OS Independent',

    ],

    python_requires='>=3.6',

    install_requires=[

        # 列出你的库依赖的其他Python包

        'requests',

        'numpy',

    ],

)