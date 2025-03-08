# CLI-AGENT

```
 ________   ___        ___                             ________   ________   _______    ________    _________   
|\   ____\ |\  \      |\  \                           |\   __  \ |\   ____\ |\  ___ \  |\   ___  \ |\___   ___\ 
\ \  \___| \ \  \     \ \  \        ____________      \ \  \|\  \\ \  \___| \ \   __/| \ \  \\ \  \\|___ \  \_| 
 \ \  \     \ \  \     \ \  \      |\____________\     \ \   __  \\ \  \  ___\ \  \_|/__\ \  \\ \  \    \ \  \  
  \ \  \____ \ \  \____ \ \  \     \|____________|      \ \  \ \  \\ \  \|\  \\ \  \_|\ \\ \  \\ \  \    \ \  \ 
   \ \_______\\ \_______\\ \__\                          \ \__\ \__\\ \_______\\ \_______\\ \__\\ \__\    \ \__\
    \|_______| \|_______| \|__|                           \|__|\|__| \|_______| \|_______| \|__| \|__|     \|__|
                                                                                                                
```

**Your AI Assistant in the Terminal**

CLI-AGENT is a command-line interface tool that provides access to AI assistants powered by Azure OpenAI and local models via vLLM. This application allows you to chat with AI models directly from your terminal, with support for web search capabilities.

## Features

- **Multiple AI Backends**:
    - Azure OpenAI integration
    - Local model support via vLLM
    - OpenAI-compatible API server option

- **Google Search Integration**:
    - Access up-to-date information via SerpAPI
    - Automatically cite sources in responses

- **User-Friendly Interface**:
    - Rich text formatting with the rich library
    - Intuitive command-line experience
    - Markdown rendering of AI responses

- **Flexibility**:
    - Choose from recommended models or specify custom ones
    - Configure via command-line arguments or interactive prompts
    - Session-based configuration management

## Prerequisites

### Required Packages

- Python 3.7 or higher

### API Keys Required

- **Azure OpenAI**: API key, endpoint, and deployment name
- **SerpAPI**: For Google search functionality (optional)

## Installation

1.  Clone this repository:
   ```bash
   git clone https://github.com/Hasan-Syed25/CLIAgent
   cd cli-agent
   ```
3.  Install the package:
   ```bash
   pip install -e .
   ```

4.  Set up your API keys (either in environment variables or via command-line arguments)

## Usage

### Basic Usage

```bash
cli-agent
```

This will launch the interactive mode that guides you through selecting a backend and providing necessary credentials.

### Command-line Options

```bash
cli-agent --backend azure --config azure-key YOUR_KEY azure-endpoint YOUR_ENDPOINT azure-deployment YOUR_DEPLOYMENT_NAME
```

```bash
cli-agent --backend vllm --model meta-llama/Llama-3-8b-instruct
```

### All Available Arguments

- `--config`: Configure API keys and settings (key-value pairs)
    - `azure-key`: Azure OpenAI API key
    - `azure-endpoint`: Azure OpenAI endpoint URL
    - `azure-deployment`: Azure OpenAI deployment name
    - `serpapi`: SerpAPI key for Google search
    - `vllm-model`: Default vLLM model name

- `--backend`: Choose the AI backend
    - `azure`: Use Azure OpenAI
    - `vllm`: Use local models via vLLM

- `--model`: Model name or path for vLLM backend
- `--tp-size`: Tensor parallel size for vLLM (default: 1)
- `--serve`: Start vLLM OpenAI-compatible server
- `--host`: Host for vLLM server (default: localhost)
- `--port`: Port for vLLM server (default: 8000)

## Supported Models

When using the vLLM backend, you can select from these recommended models:

- microsoft/Phi-4-multimodal-instruct
- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Llama-2-13b-chat-hf
- meta-llama/Llama-3-8b-instruct
- mistralai/Mistral-7B-Instruct-v0.2
- mistralai/Mixtral-8x7B-Instruct-v0.1
- Qwen/Qwen1.5-7B-Chat
- Qwen/Qwen1.5-14B-Chat
- Qwen/Qwen2.5-1.5B-Instruct
- Qwen/Qwen2.5-7B-Instruct
- stabilityai/StableBeluga2
- codellama/CodeLlama-7b-Instruct-hf
- TheBloke/Llama-2-7B-Chat-GGUF
- microsoft/phi-2

You can also specify a custom model path.

## Environment Variables

The following environment variables can be used instead of command-line arguments:

- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL
- `AZURE_OPENAI_DEPLOYMENT`: Azure OpenAI deployment name
- `SERPAPI_API_KEY`: SerpAPI key for Google search functionality

## Examples

### Azure OpenAI with Google Search

```bash
cli-agent --backend azure --config azure-key YOUR_KEY azure-endpoint YOUR_ENDPOINT azure-deployment YOUR_DEPLOYMENT --config serpapi YOUR_SERPAPI_KEY
```

### Running Local Models with vLLM

```bash
cli-agent --backend vllm --model meta-llama/Llama-3-8b-instruct
```

## Agent Capabilities

### Azure OpenAI Agent

- Uses official Azure OpenAI API
- Function calling for tools like Google Search
- Maintains conversation context
- Cites sources when using search results

### vLLM Agent

- Uses local models via vLLM
- Provides OpenAI-compatible API access
- Supports a wide range of open-source models
- Efficiently manages model resources

## Troubleshooting

### Common Issues

1.  **Azure OpenAI connection failed**:
        - Verify your API key, endpoint, and deployment name
        - Check network connectivity to Azure services

2.  **vLLM model loading issues**:
        - Ensure you have enough GPU memory for the selected model
        - Check that the model path or name is correct
        - Verify vLLM is installed correctly with `pip install vllm`

3.  **SerpAPI search not working**:
        - Confirm your SerpAPI key is valid
        - Check your search query formatting

### Getting Help

If you encounter issues not covered here, please open an issue on the GitHub repository with:

- The command you were trying to run
- The complete error message
- Your environment details (OS, Python version, package versions)

## Acknowledgments

- This project uses the OpenAI and Azure OpenAI APIs
- vLLM for efficient local model inference
- Rich library for terminal formatting
- SerpAPI for Google search functionality
