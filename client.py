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

        self.model_name = config.get("moodle:mcp", "gemini-2.0-flash")

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
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self):
        try:
            servers = config.get("servers", {})
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    async def process_query(self, query: str) -> str:
        self.messages.append({"role": "user", "content": query})

        try:
            gemini_tools = [
                types.Tool(
                    function_declarations=[
                        {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": {
                                k: v for k, v in tool["input_schema"].items()
                                if k not in ["additionalProperties", "$schema"]
                            },
                        }
                    ]
                )
                for tool in self.available_tools
            ]

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
                print(f"Tool call detected: {function_call.name}")

                session = self.tool_to_session.get(function_call.name, self.sessions[0])
                result = await session.call_tool(function_call.name, arguments=dict(function_call.args))

                raw = result.content[0].text if result.content else str(result)
                try:
                    parsed = json.loads(raw)
                    tool_result = json.dumps(parsed, indent=2)
                except Exception:
                    tool_result = raw

                print("--- Tool Result ---")
                print(tool_result)

                self.messages.append({"role": "tool_result", "content": tool_result})
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=f"Tool result: {tool_result}")]))

                final_response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=0),
                )
                text = final_response.text
            else:
                text = response.text

            self.messages.append({"role": "assistant", "content": text})
            print("--- AI Response ---")
            print(text)
            return text

        except Exception as e:
            traceback.print_exc()
            error_msg = f"Error when calling the model: {e}"
            print(error_msg)
            return error_msg

    async def chat_loop(self):
        print("\nMCP Chatbot with Gemini Started!")
        print("Type your queries or 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                await self.process_query(query)
            except Exception as e:
                print(f"Error in chat loop: {e}")
                traceback.print_exc()

    async def run(self):
        async with self.exit_stack:
            await self.connect_to_servers()
            await self.chat_loop()


async def main():
    try:
        print("Starting MCP Chatbot...")
        chatbot = MCP_ChatBot()
        await chatbot.run()
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
