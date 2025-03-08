import os
import sys
import json
import argparse
import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt, IntPrompt
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from dotenv import load_dotenv
import asyncio
import aiohttp

# Load environment variables
load_dotenv("C:/Users/hasan/OneDrive/Desktop/Azure OpenAI Agent/azure_oai_agent/.env")

console = Console()

# Safely import OpenAI
try:
    from openai import AzureOpenAI, OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    console.print("[bold yellow]OpenAI SDK not installed. To use OpenAI or Azure OpenAI, install with: pip install openai[/]")

# Safely import vLLM if available
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    
except ImportError:
    VLLM_AVAILABLE = False
    console.print("[bold yellow]vLLM not installed. To use vLLM models, install with: pip install vllm[/]")

class GoogleSearchTool:
    """Google Search tool using SerpAPI"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            console.print("[bold yellow]No SerpAPI key found. Search functionality will be limited.[/]")

    def search(self, query, num_results=5):
        """Perform a Google search and return results"""
        if not self.api_key:
            return {"error": "SerpAPI key not configured. Use 'config --serpapi YOUR_KEY' to set it."}
            
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": num_results
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            results = response.json()
            
            # Extract organic results
            if "organic_results" in results:
                return {
                    "results": [
                        {
                            "title": result.get("title", ""),
                            "link": result.get("link", ""),
                            "snippet": result.get("snippet", "")
                        }
                        for result in results["organic_results"][:num_results]
                    ]
                }
            else:
                return {"error": "No results found"}
                
        except requests.exceptions.RequestException as e:
            return {"error": f"Search request failed: {str(e)}"}
        except json.JSONDecodeError:
            return {"error": "Failed to parse search results"}

class AzureOpenAIAgent:
    """Azure OpenAI-powered agent with tool access"""
    
    def __init__(self, api_key=None, endpoint=None, deployment=None, config_manager=None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI SDK not installed. Please install with: pip install openai")
            
        self.api_key = config_manager.get("azure_openai_api_key") or api_key
        self.endpoint = config_manager.get("azure_openai_endpoint") or endpoint
        self.deployment = config_manager.get("azure_openai_deployment") or deployment
        
        if not self.api_key or not self.endpoint or not self.deployment:
            console.print("[bold red]Azure OpenAI configuration is incomplete. Please provide API key, endpoint, and deployment.[/]")
            sys.exit(1)
            
        try:
            self.client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                azure_deployment=self.deployment,
                api_key=self.api_key,
                api_version="2024-08-01-preview"
            )
            # Test the connection
            console.print("[dim]Testing Azure OpenAI connection...[/dim]")
            self.client.models
            console.print("[green]Azure OpenAI connection successful![/green]")
        except Exception as e:
            console.print(f"[bold red]Failed to initialize Azure OpenAI client: {str(e)}[/]")
            sys.exit(1)
        
        serp_api_key = config_manager.get("serpapi_api_key")
        self.google_search = GoogleSearchTool(api_key=serp_api_key)
        
        # Agent system prompt
        self.system_prompt = """You are a helpful AI assistant with access to Google Search.
When you need information you don't know, you can use the google_search tool.
Always provide thoughtful, accurate responses based on the most up-to-date information available to you.
If you use search results, cite your sources."""
        
        self.messages = [{"role": "system", "content": self.system_prompt}]
    
    def _run_google_search(self, query):
        """Run a Google search and format results"""
        search_results = self.google_search.search(query)
        
        if "error" in search_results:
            return f"Error performing search: {search_results['error']}"
            
        formatted_results = "Search Results:/n/n"
        for i, result in enumerate(search_results.get("results", []), 1):
            formatted_results += f"{i}. {result['title']}/n"
            formatted_results += f"   URL: {result['link']}/n"
            formatted_results += f"   {result['snippet']}/n/n"
            
        return formatted_results
    
    def _handle_tool_calls(self, tool_calls):
        """Handle tool calls from the Azure OpenAI API"""
        tool_results = []
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name == "google_search":
                query = function_args.get("query", "")
                console.print(f"[dim]Searching Google for: {query}[/dim]")
                result = self._run_google_search(query)
                
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": result
                })
                
        return tool_results
    
    def chat(self, user_input):
        """Process a user message and return agent response"""
        # Add user message to conversation
        self.messages.append({"role": "user", "content": user_input})
        
        try:
            # Get response from Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=self.messages,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "google_search",
                            "description": "Search Google for information",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The search query to send to Google"
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    }
                ],
                tool_choice="auto"
            )
            
            # Get the message from the response
            response_message = response.choices[0].message
            
            # Handle any tool calls
            if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                # Show that the agent is using tools
                console.print("[dim]Using tools to find information...[/dim]")
                
                # Process tool calls and get results
                tool_results = self._handle_tool_calls(response_message.tool_calls)
                
                # Add the assistant's message with tool calls to the messages
                self.messages.append(response_message)
                
                # Add tool results to messages
                self.messages.extend(tool_results)
                
                # Get a new response from the assistant with the tool results
                second_response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=self.messages
                )
                
                assistant_response = second_response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": assistant_response})
            else:
                # No tool calls, just get the content
                assistant_response = response_message.content
                self.messages.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            console.print(f"[bold red]{error_msg}[/]")
            # Add a fallback response
            fallback_response = "I'm sorry, I encountered an error while processing your request. Please try again or check your Azure OpenAI configuration."
            self.messages.append({"role": "assistant", "content": fallback_response})
            return fallback_response

class OpenAICompatibleVLLMAgent:
    """vLLM agent that uses the OpenAI-compatible API via Python client"""

    RECOMMENDED_MODELS = [
        "microsoft/Phi-4-multimodal-instruct",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-3-8b-instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen/Qwen1.5-7B-Chat",
        "Qwen/Qwen1.5-14B-Chat",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "stabilityai/StableBeluga2",
        "codellama/CodeLlama-7b-Instruct-hf",
        "TheBloke/Llama-2-7B-Chat-GGUF",
        "microsoft/phi-2"
    ]
    
    def __init__(self, base_url="http://localhost:8000/v1", model_name=None, config_manager=None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI SDK not installed. Please install with: pip install openai")
        
        self.base_url = base_url
        self.model_name = model_name  # This will be fetched from the server if None
        self.client = OpenAI(base_url=base_url, api_key="dummy")
        
        # System prompt
        self.system_prompt = """You are a helpful AI assistant.
Always provide thoughtful, accurate responses based on the most up-to-date information available to you.
If you don't know something, just say so."""
        
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.google_search = GoogleSearchTool(config_manager.get("serpapi_api_key"))
        
        
    def chat(self, user_input):
        """Process a user message and return agent response"""
        # Add user message to conversation
        self.messages.append({"role": "user", "content": user_input})
        
        try: 
            # Get response from vLLM OpenAI-compatible server
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
            )
            
            # Get the message from the response
            response_message = response.choices[0].message
            assistant_response = response_message.content
            self.messages.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            console.print(f"[bold red]{error_msg}[/]")
            # Add a fallback response
            fallback_response = "I'm sorry, I encountered an error while processing your request. Please try again."
            self.messages.append({"role": "assistant", "content": fallback_response})
            return fallback_response

# Global variable to store the subprocess instance
server_process_instance = None

async def start_vllm_server(model_name, host="localhost", port=8000, tensor_parallel_size=1):
    """Start a vLLM OpenAI-compatible server in a separate process using asyncio,
       and wait until it is completely available.
    """
    global server_process_instance

    console.print(f"[bold blue]Starting vLLM OpenAI-compatible server with model {model_name}...[/]")

    # Construct the command with required arguments
    command = [
        "vllm", "serve", f"{model_name}"
    ]

    console.print(f"[dim]Running command: {' '.join(command)}[/dim]")
    server_process_instance = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        text=False
    )

    # Function to asynchronously stream output
    async def stream_reader(stream, prefix):
        while True:
            line = await stream.readline()
            if line:
                print(f"{prefix}{line.strip()}")
            else:
                break

    # Launch background tasks to read stdout and stderr
    asyncio.create_task(stream_reader(server_process_instance.stdout, "[dim]"))
    asyncio.create_task(stream_reader(server_process_instance.stderr, "[red]"))

    console.print("[dim]Waiting for server to become available...[/dim]")
    url = f"http://{host}:{port}/v1/models"
    max_attempts = 120  # Wait up to 2 minutes (120 seconds)
    attempts = 0

    async with aiohttp.ClientSession() as session:
        while attempts < max_attempts:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        console.print(f"[green]vLLM OpenAI-compatible server started successfully at http://{host}:{port}[/green]")
                        return f"http://{host}:{port}/v1"
            except aiohttp.ClientConnectionError:
                pass  # Server not up yet
            attempts += 1
            await asyncio.sleep(1)

    console.print("[bold yellow]Server might still be starting. Will continue anyway.[/yellow]")
    return f"http://{host}:{port}/v1"

async def stop_vllm_server():
    """Stop the running vLLM server subprocess asynchronously."""
    global server_process_instance
    if server_process_instance:
        console.print("[bold red]Stopping vLLM server...[/bold red]")
        server_process_instance.terminate()
        await server_process_instance.wait()  # Wait for process to terminate
        console.print("[green]vLLM server stopped.[/green]")
        server_process_instance = None
    else:
        console.print("[bold yellow]No vLLM server process is running.[/yellow]")

class ConfigManager:
    """Manage API keys and configuration in-memory for a single session."""

    def __init__(self):
        self.config = {}

    def get(self, key, default=None):
        """Get a configuration value"""
        return self.config.get(key, default)

    def set(self, key, value):
        """Set a configuration value"""
        self.config[key] = value
        return True

def display_welcome_message():
    """Display fancy welcome message with pixelated robot face and CLI-AGENT text"""
    console = Console()
    
    welcome_text = Text(r"""
 ________   ___        ___                             ________   ________   _______    ________    _________   
|\   ____\ |\  \      |\  \                           |\   __  \ |\   ____\ |\  ___ \  |\   ___  \ |\___   ___\ 
\ \  \___| \ \  \     \ \  \        ____________      \ \  \|\  \\ \  \___| \ \   __/| \ \  \\ \  \\|___ \  \_| 
 \ \  \     \ \  \     \ \  \      |\____________\     \ \   __  \\ \  \  ___\ \  \_|/__\ \  \\ \  \    \ \  \  
  \ \  \____ \ \  \____ \ \  \     \|____________|      \ \  \ \  \\ \  \|\  \\ \  \_|\ \\ \  \\ \  \    \ \  \ 
   \ \_______\\ \_______\\ \__\                          \ \__\ \__\\ \_______\\ \_______\\ \__\\ \__\    \ \__\
    \|_______| \|_______| \|__|                           \|__|\|__| \|_______| \|_______| \|__| \|__|     \|__|
                                                                                                                
                                                                                                                
                                                                                                                
""")
    
    # Print the welcome text centered in the terminal
    console.print(Align.center(welcome_text), style="bold green")
    console.print(Align.center(Text("Your AI Assistant in the Terminal", style="bold blue")))
    console.print("\n")

def select_vllm_model():
    """Allow user to select a vLLM model from a list"""
    console.print("[bold cyan]Choose a model to use with vLLM:[/]")
    
    for i, model in enumerate(OpenAICompatibleVLLMAgent.RECOMMENDED_MODELS, 1):
        console.print(f"  {i}. [yellow]{model}[/]")
    
    console.print(f"  {len(OpenAICompatibleVLLMAgent.RECOMMENDED_MODELS) + 1}. [green]Enter a custom model name[/]")
    
    while True:
        try:
            choice = IntPrompt.ask(
                "[bold cyan]Enter your choice[/]", 
                default=1,
                show_choices=False
            )
            
            if 1 <= choice <= len(OpenAICompatibleVLLMAgent.RECOMMENDED_MODELS):
                return OpenAICompatibleVLLMAgent.RECOMMENDED_MODELS[choice - 1]
            elif choice == len(OpenAICompatibleVLLMAgent.RECOMMENDED_MODELS) + 1:
                return Prompt.ask("[bold cyan]Enter the model name/path[/]")
            else:
                console.print("[yellow]Invalid choice. Please try again.[/]")
        except ValueError:
            console.print("[yellow]Please enter a valid number.[/]")

async def run_vllm_chat(agent):
    """Run the chat loop with vLLM backend"""
    try:
        while True:
            user_input = Prompt.ask("[bold green]You")
            
            if user_input.lower() in ('exit', 'quit'):
                console.print("[bold blue]Goodbye![/]")
                break
                
            # Display thinking indicator
            with console.status("[bold blue]Thinking...[/]"):
                response = await agent.chat(user_input)
            
            # Print the response
            console.print("[bold blue]Assistant")
            console.print(Markdown(response))
            console.print()
            
    except KeyboardInterrupt:
        console.print("/n[bold blue]Goodbye![/]")
    except Exception as e:
        console.print(f"/n[bold red]An error occurred: {str(e)}[/]")
        console.print("[bold blue]Exiting...[/]")

def run_openai_compatible_vllm_chat(agent):
    """Run the chat loop with OpenAI-compatible vLLM backend"""
    try:
        while True:
            user_input = Prompt.ask("[bold green]You")
            
            if user_input.lower() in ('exit', 'quit'):
                console.print("[bold blue]Goodbye![/]")
                break
                
            # Display thinking indicator
            with console.status("[bold blue]Thinking...[/]"):
                response = agent.chat(user_input)
            
            # Print the response
            console.print("[bold blue]Assistant")
            console.print(Markdown(response))
            console.print()
            
    except KeyboardInterrupt:
        console.print("/n[bold blue]Goodbye![/]")
    except Exception as e:
        console.print(f"/n[bold red]An error occurred: {str(e)}[/]")
        console.print("[bold blue]Exiting...[/]")

def run_azure_chat(agent):
    """Run the chat loop with Azure OpenAI backend"""
    try:
        while True:
            user_input = Prompt.ask("[bold green]You")
            
            if user_input.lower() in ('exit', 'quit'):
                console.print("[bold blue]Goodbye![/]")
                break
                
            # Display thinking indicator
            with console.status("[bold blue]Thinking...[/]"):
                response = agent.chat(user_input)
            
            # Print the response
            console.print("[bold blue]Assistant")
            console.print(Markdown(response))
            console.print()
            
    except KeyboardInterrupt:
        console.print("/n[bold blue]Goodbye![/]")
    except Exception as e:
        console.print(f"/n[bold red]An error occurred: {str(e)}[/]")
        console.print("[bold blue]Exiting...[/]")

async def entry_point():
    parser = argparse.ArgumentParser(description="AI Assistant CLI with Azure OpenAI and vLLM support")
    parser.add_argument("--config", nargs="+", help="Configure API keys (--config azure-key YOUR_KEY, --config azure-endpoint YOUR_ENDPOINT, --config azure-deployment YOUR_DEPLOYMENT_NAME, or --config serpapi YOUR_KEY)")
    parser.add_argument("--backend", choices=["azure", "vllm", "vllm-server"], help="Choose the backend (azure, vllm, or vllm-server)")
    parser.add_argument("--model", help="Model name or path for vLLM backend")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--serve", action="store_true", help="Start vLLM OpenAI-compatible server")
    parser.add_argument("--host", default="localhost", help="Host for vLLM server")
    parser.add_argument("--port", type=int, default=8000, help="Port for vLLM server")
    args = parser.parse_args()
    
    config_manager = ConfigManager()
    
    # Handle configuration from key-value pairs in args.config
    if args.config:
        if len(args.config) % 2 != 0:
            console.print("[bold red]Configuration arguments should be provided as key value pairs.[/]")
            return
        for i in range(0, len(args.config), 2):
            key_type = args.config[i].lower()
            value = args.config[i+1]

            if key_type == "azure-key":
                config_manager.set("azure_openai_api_key", value)
                console.print("[green]Azure OpenAI API key configured successfully.[/]")
            elif key_type == "azure-endpoint":
                config_manager.set("azure_openai_endpoint", value)
                console.print("[green]Azure OpenAI endpoint configured successfully.[/]")
            elif key_type == "azure-deployment":
                config_manager.set("azure_openai_deployment", value)
                console.print("[green]Azure OpenAI deployment name configured successfully.[/]")
            elif key_type == "serpapi":
                config_manager.set("serpapi_api_key", value)
                console.print("[green]SerpAPI key configured successfully.[/]")
            elif key_type == "vllm-model":
                config_manager.set("vllm_model", value)
                console.print("[green]Default vLLM model configured successfully.[/]")
            else:
                console.print(f"[bold red]Unknown configuration key: {key_type}[/]")
                return
    
    # Ask user to choose a backend if not specified
    backend = args.backend
    available_backends = []
    
    if OPENAI_AVAILABLE:
        available_backends.append(("azure", "Azure OpenAI"))
    if VLLM_AVAILABLE:
        available_backends.append(("vllm", "vLLM (local models)"))
    
    if not available_backends:
        console.print("[bold red]No AI backends available. Please install either the OpenAI or vLLM package.[/]")
        return
    
    if not backend:
        console.print("[bold cyan]Choose a backend:[/]")
        for i, (backend_id, backend_name) in enumerate(available_backends, 1):
            console.print(f"  {i}. {backend_name}")
        
        while True:
            try:
                choice = IntPrompt.ask(
                    "[bold cyan]Enter your choice[/]", 
                    default=1,
                    show_choices=False
                )
                
                if 1 <= choice <= len(available_backends):
                    backend = available_backends[choice-1][0]
                    break
                else:
                    console.print("[yellow]Invalid choice. Please try again.[/]")
            except ValueError:
                console.print("[yellow]Please enter a valid number.[/]")
    
    # Check for SerpAPI key in all backends
    serpapi_api_key = config_manager.get("serpapi_api_key") or os.getenv("SERPAPI_API_KEY")
    if not serpapi_api_key:
        serpapi_api_key = Prompt.ask("[bold yellow]Please enter your SerpAPI key (or press Enter to skip)[/]", password=True)
        if serpapi_api_key:
            config_manager.set("serpapi_api_key", serpapi_api_key)
    
    # Handle chosen backend
    if backend == "azure":
        if not OPENAI_AVAILABLE:
            console.print("[bold red]Azure OpenAI SDK not installed. Please install with: pip install openai[/]")
            return
            
        # Setup Azure OpenAI
        azure_api_key = config_manager.get("azure_openai_api_key") or os.getenv("AZURE_OPENAI_API_KEY")
        if not azure_api_key:
            azure_api_key = Prompt.ask("[bold yellow]Please enter your Azure OpenAI API key[/]", password=True)
            config_manager.set("azure_openai_api_key", azure_api_key)
        
        azure_endpoint = config_manager.get("azure_openai_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            azure_endpoint = Prompt.ask("[bold yellow]Please enter your Azure OpenAI endpoint URL[/]")
            config_manager.set("azure_openai_endpoint", azure_endpoint)
        
        azure_deployment = config_manager.get("azure_openai_deployment") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not azure_deployment:
            azure_deployment = Prompt.ask("[bold yellow]Please enter your Azure OpenAI deployment name[/]")
            config_manager.set("azure_openai_deployment", azure_deployment)

        try:
            # Initialize agent
            agent = AzureOpenAIAgent(
                api_key=azure_api_key,
                endpoint=azure_endpoint,
                deployment=azure_deployment,
                config_manager=config_manager
            )
            
            # Display chat panel
            console.print(Panel.fit(
                "[bold blue]Azure OpenAI Agent with Google Search[/]\n" +
                "[dim]Type 'exit', 'quit', or press Ctrl+C to exit the program[/]",
                title="Chat Session"
            ))
            
            # Run chat with Azure backend
            run_azure_chat(agent)
        except Exception as e:
            console.print(f"[bold red]Failed to start Azure OpenAI chat: {str(e)}[/]")
    
    elif backend == "vllm":
        if not VLLM_AVAILABLE:
            console.print("[bold red]vLLM server components not available. Please install with: pip install vllm[/]")
            return
            
        try:
            # Get model name
            model_name = args.model or config_manager.get("vllm_model")
            if not model_name:
                model_name = select_vllm_model()
                # Save as default for future use
                config_manager.set("vllm_model", model_name)
            
            # Start server if requested
            base_url = await start_vllm_server(
                model_name=model_name
            )
            if not base_url:
                return
            
            # Initialize OpenAI-compatible agent
            agent = OpenAICompatibleVLLMAgent(
                base_url=base_url,
                model_name=model_name,
                config_manager=config_manager
            )
            
            # Display chat panel
            console.print(Panel.fit(
                f"[bold blue]vLLM OpenAI-compatible Agent with {model_name}[/]/n" +
                "[dim]Type 'exit', 'quit', or press Ctrl+C to exit the program[/]",
                title="Chat Session"
            ))
            
            # Run chat loop
            run_openai_compatible_vllm_chat(agent)
        except Exception as e:
            console.print(f"[bold red]Failed to start vLLM OpenAI-compatible chat: {str(e)}[/]")
    else:
        console.print("[bold red]Unknown backend selected. Please choose either 'azure', 'vllm', or 'vllm-server'[/]")
        return
    
    console.print("[bold blue]Exiting...[/]")

def main():
    # This ensures that the async code is properly awaited.
    display_welcome_message()
    asyncio.run(entry_point())

if __name__ == '__main__':
    main()