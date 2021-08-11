from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="torchRL",
        version="0.1",
        author="Joonil Ahn",
        author_email="joonilahn1@gmail.com",
        packages=find_packages(),
        install_requires=[
            "torch>=1.6.0",
            "numpy>=1.21.1",
            "gym==0.18.3",
            "yacs==0.1.8",
        ],
    )
