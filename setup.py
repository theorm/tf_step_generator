import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tf-step-generator", # Replace with your own username
    version="0.0.1",
    author="Roman Kalyakin",
    author_email="roman@kalyakin.com",
    description="A flexible text generator for Huggingface Transfomrers language models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/tf-step-generator",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'transformers>=3.1.0,<5.0.0',
        'torch>=1.6.0',
        'numpy',
    ],
)