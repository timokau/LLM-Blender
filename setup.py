from setuptools import setup, find_packages
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='llm_blender',
    version='0.0.2',
    description='LLM-Blender, an innovative ensembling framework to attain consistently superior performance by leveraging the diverse strengths and weaknesses of multiple open-source large language models (LLMs). LLM-Blender cut the weaknesses through ranking and integrate the strengths through fusing generation to enhance the capability of LLMs.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Dongfu Jiang',
    author_email='dongfu.jdf@gmail.com',
    packages=find_packages(),
    url='https://yuchenlin.xyz/LLM-Blender/',
    entry_points={
        'console_scripts': [
            'train_ranker = llm_blender.train_ranker:main',
        ],
    },
    install_requires=[
        'transformers',
        'torch',
        'numpy',
        'accelerate',
        'safetensors',
        'dataclasses-json',
        'sentencepiece',
        'protobuf',
    ],
    extras_require={
        "example": [
            'datasets',
            'scipy',
            'jupyter'
        ],
        "train": [
            'datasets',
            'bitsandbytes',
            'deepspeed',
            'wandb',
        ],
        "data": [
            'datasets',
            'openai',
            'peft',
            'fschat',
        ],
        "eval": [
            'datasets',
            'pycocoevalcap',
            'spacy',
            'prettytable',
            'evaluate',
            'bert_score',
            'tabulate',
            'scipy',
            'nltk',
            'scikit-learn',
            'sacrebleu',
            'rouge_score',
        ],
    },
)
