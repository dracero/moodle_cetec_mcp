from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
import nest_asyncio
import os
import json
import traceback
from google import genai
from google.genai import types
from contextlib import AsyncExitStack

nest_asyncio.apply()
load_dotenv()

# Load configuration file
CONFIG_PATH = os.environ.get("MCP_CONFIG_PATH", "mcp_config.json")
try:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
except Exception as e:
    print(f"ERROR: Failed to load config file: {e}")
    raise

class MCP_ChatBot:
    def __init__(self):
        self.sessions: List[ClientSession] = []
        self.tool_to_session = {}
        self.exit_stack = AsyncExitStack()

        self.model_name = config.get("model", "gemini-2.0-flash")

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("ERROR: GOOGLE_API_KEY not found in environment variables!")
            print("Please add your Google API key to .env file")
            return

        try:
            self.client = genai.Client(api_key=api_key)
            print(f"Gemini client initialized successfully with model: {self.model_name}")
        except Exception as e:
            print(f"Error initializing Gemini client: {e}")
            traceback.print_exc()
            raise

        self.available_tools: List[dict] = []
        self.messages: List[dict] = []

    def clean_schema(self, schema: dict) -> dict:
        """Clean schema to remove properties that cause validation errors in Gemini."""
        if not isinstance(schema, dict):
            return schema
        
        cleaned = {}
        for key, value in schema.items():
            if key in ["additionalProperties", "$schema"]:
                continue
            
            if isinstance(value, dict):
                if key == "properties":
                    # Clean each property recursively
                    cleaned[key] = {}
                    for prop_key, prop_value in value.items():
                        cleaned[key][prop_key] = self.clean_schema(prop_value)
                else:
                    cleaned[key] = self.clean_schema(value)
            elif isinstance(value, list):
                cleaned[key] = [self.clean_schema(item) if isinstance(item, dict) else item for item in value]
            else:
                cleaned[key] = value
        
        return cleaned

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions.append(session)

            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])

            for tool in tools:
                self.tool_to_session[tool.name] = session
                cleaned_schema = self.clean_schema(tool.inputSchema)
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": cleaned_schema
                })
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")
            traceback.print_exc()

    async def connect_to_servers(self):
        try:
            servers = config.get("servers", {})
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    def print_server_response_details(self, result):
        """Print detailed information about the server response"""
        print("\n" + "="*50)
        print("DETAILED SERVER RESPONSE")
        print("="*50)
        
        # Print the complete result object
        print(f"Result type: {type(result)}")
        print(f"Result attributes: {dir(result)}")
        
        # Print content if available
        if hasattr(result, 'content') and result.content:
            print(f"\nContent count: {len(result.content)}")
            for i, content_item in enumerate(result.content):
                print(f"\nContent {i+1}:")
                print(f"  Type: {type(content_item)}")
                print(f"  Attributes: {dir(content_item)}")
                
                if hasattr(content_item, 'text'):
                    print(f"  Text: {content_item.text}")
                if hasattr(content_item, 'type'):
                    print(f"  Content Type: {content_item.type}")
                    
                # Print all attributes of the content item
                for attr in dir(content_item):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(content_item, attr)
                            if not callable(value):
                                print(f"  {attr}: {value}")
                        except:
                            pass
        
        # Print metadata if available
        if hasattr(result, 'meta'):
            print(f"\nMetadata: {result.meta}")
            
        # Print error information if available
        if hasattr(result, 'isError') and result.isError:
            print(f"\nError detected: {result.isError}")
            
        # Print the complete result as string
        print(f"\nComplete result as string:")
        print(str(result))
        
        # Try to print as JSON if possible
        try:
            print(f"\nResult as dict/JSON:")
            if hasattr(result, '__dict__'):
                print(json.dumps(result.__dict__, indent=2, default=str))
        except Exception as e:
            print(f"Could not convert to JSON: {e}")
            
        print("="*50)

    async def process_query(self, query: str) -> str:
        self.messages.append({"role": "user", "content": query})

        try:
            gemini_tools = []
            for tool in self.available_tools:
                try:
                    gemini_tool = types.Tool(
                        function_declarations=[
                            {
                                "name": tool["name"],
                                "description": tool["description"],
                                "parameters": tool["input_schema"],
                            }
                        ]
                    )
                    gemini_tools.append(gemini_tool)
                except Exception as e:
                    print(f"Warning: Failed to add tool {tool['name']}: {e}")
                    # Continue with other tools
                    continue

            contents: List[types.Content] = []
            for msg in self.messages:
                if msg["role"] == "user":
                    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg["content"])]))
                elif msg["role"] == "assistant":
                    contents.append(types.Content(role="model", parts=[types.Part.from_text(text=msg["content"])]))
                elif msg["role"] == "tool_result":
                    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=f"Tool result: {msg['content']}")]))

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=gemini_tools,
                ),
            )

            candidate = response.candidates[0]
            if candidate.content.parts and getattr(candidate.content.parts[0], 'function_call', None):
                function_call = candidate.content.parts[0].function_call
                print(f"\n🔧 Tool call detected: {function_call.name}")
                print(f"📝 Arguments: {dict(function_call.args)}")

                session = self.tool_to_session.get(function_call.name, self.sessions[0])
                
                print(f"\n🚀 Calling MCP server...")
                result = await session.call_tool(function_call.name, arguments=dict(function_call.args))

                # Print detailed server response
                self.print_server_response_details(result)

                # Extract and format the response
                if result.content:
                    # Collect all content parts
                    all_content = []
                    for content_item in result.content:
                        if hasattr(content_item, 'text') and content_item.text:
                            all_content.append(content_item.text)
                        elif hasattr(content_item, 'data'):
                            all_content.append(str(content_item.data))
                        else:
                            all_content.append(str(content_item))
                    
                    raw_response = '\n'.join(all_content) if all_content else str(result)
                else:
                    raw_response = str(result)

                # Try to format as JSON if possible
                try:
                    parsed = json.loads(raw_response)
                    tool_result = json.dumps(parsed, indent=2)
                    print(f"\n📄 Formatted Tool Result (JSON):")
                    print(tool_result)
                except Exception:
                    tool_result = raw_response
                    print(f"\n📄 Raw Tool Result:")
                    print(tool_result)

                self.messages.append({"role": "tool_result", "content": tool_result})
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=f"Tool result: {tool_result}")]))

                print(f"\n🤖 Generating final response...")
                final_response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=0),
                )
                text = final_response.text
            else:
                text = response.text

            self.messages.append({"role": "assistant", "content": text})
            print("\n" + "🎯 AI FINAL RESPONSE " + "="*30)
            print(text)
            print("="*50)
            return text

        except Exception as e:
            traceback.print_exc()
            error_msg = f"Error when calling the model: {e}"
            print(error_msg)
            return error_msg

    async def chat_loop(self):
        print("\n🚀 MCP Chatbot with Gemini Started!")
        print("💬 Type your queries or 'quit' to exit.")
        print("📊 This version shows complete server responses!")
        
        while True:
            try:
                query = input("\n🔍 Query: ").strip()
                if query.lower() == 'quit':
                    break
                await self.process_query(query)
            except Exception as e:
                print(f"❌ Error in chat loop: {e}")
                traceback.print_exc()

    async def run(self):
        async with self.exit_stack:
            await self.connect_to_servers()
            await self.chat_loop()


async def main():
    try:
        print("🔄 Starting MCP Chatbot...")
        chatbot = MCP_ChatBot()
        await chatbot.run()
    except Exception as e:
        print(f"❌ Error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())