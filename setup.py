from setuptools import setup, find_packages

setup(
    name="azure_oai_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai",
        "requests",
        "rich",
        "python-dotenv",
        "aiohttp"
    ],
    entry_points={
        "console_scripts": [
            "cli-agent=azure_oai_agent.cli:main",
        ],
    },
    author="Syed Hasan",
    author_email="hasansyed8505@gmail.com",
    description="Azure OpenAI Agent with Google Search CLI",
    keywords="azure, openai, cli, agent, google search",
    python_requires=">=3.7",
)