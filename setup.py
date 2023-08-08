from pathlib import Path

from setuptools import find_packages, setup

if __name__ == "__main__":
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    REQUIREMENTS = [
        "einops",
        "einops-exts",
        "transformers>=4.28.1",
        "torch==2.0.1",
        "pillow",
        "open_clip_torch>=2.16.0",
        "sentencepiece==0.1.98"
    ]

    DEV = [
        "wandb",
        "torchvision",
        "nltk",
        "inflection",
        "pycocoevalcap",
        "pycocotoolsa",
    ]

    TRAINING = [
        "torchvision",
        "braceexpand",
        "tqdm",
        "webdataset",
    ]

    setup(
        name="open_flamingo",
        packages=find_packages(),
        include_package_data=True,
        version="2.0.0",
        license="MIT",
        description="An open-source framework for training large multimodal models",
        long_description=long_description,
        long_description_content_type="text/markdown",
        data_files=[(".", ["README.md"])],
        keywords=["machine learning"],
        install_requires=REQUIREMENTS,
        extras_require={
            "dev": DEV,
            "training": TRAINING,
            "all": list(set(REQUIREMENTS + DEV + TRAINING)),
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.9",
        ],
    )
