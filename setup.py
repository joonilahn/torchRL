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
            "tensorboard",
            "numpy",
            "gym==0.17.0",
            "atari-py==0.2.5",
            "pyglet==1.5.11",
            "yacs",
        ],
    )
