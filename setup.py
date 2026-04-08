from setuptools import setup, find_packages

setup(
    name='llm_cascade',
    version='0.1.0',
    description='Automatic fallback across 8 free-tier LLM providers',
    author='KarAnalytics',
    url='https://github.com/KarAnalytics/llm_cascade',
    packages=find_packages(),
    install_requires=[
        'openai',
        'google-genai',
    ],
    python_requires='>=3.8',
)
