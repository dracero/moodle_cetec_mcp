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

nest_asyncio.apply()
load_dotenv()

class MCP_ChatBot:
    def __init__(self):
        # Initialize session and client objects
        self.session: ClientSession = None
        
        # Check if API key exists
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("ERROR: GOOGLE_API_KEY not found in environment variables!")
            print("Please add your Google API key to .env file")
            return
        
        # Initialize the Gemini client directly
        try:
            self.client = genai.Client(api_key=api_key)
            self.model_name = "gemini-2.0-flash"
            print(f"Gemini client initialized successfully with model: {self.model_name}")
        except Exception as e:
            print(f"Error initializing Gemini client: {e}")
            traceback.print_exc()
            raise
        
        self.available_tools: List[dict] = []
        self.messages: List[dict] = []

    async def process_query(self, query: str) -> str:
        """
        Sends query to Gemini model with available tools,
        handles tool calls, and returns the final text response.
        """
        # Add user message to the history
        self.messages.append({"role": "user", "content": query})

        try:
            # Prepare tools schema compatible with Gemini
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

            # Build contents using SDK types.Content and types.Part
            contents: List[types.Content] = []
            for msg in self.messages:
                if msg["role"] == "user":
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=msg["content"])],
                        )
                    )
                elif msg["role"] == "assistant":
                    contents.append(
                        types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=msg["content"])],
                        )
                    )
                elif msg["role"] == "tool_result":
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=f"Tool result: {msg['content']}")],
                        )
                    )

            # First call to Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=gemini_tools,
                ),
            )

            # Check for function call
            candidate = response.candidates[0]
            if candidate.content.parts and getattr(candidate.content.parts[0], 'function_call', None):
                function_call = candidate.content.parts[0].function_call
                print(f"Tool call detected: {function_call.name}")

                # Execute the tool via MCP session
                result = await self.session.call_tool(
                    function_call.name,
                    arguments=dict(function_call.args),
                )

                # Extract and format result
                raw = None
                try:
                    raw = result.content[0].text
                except Exception:
                    raw = str(result)

                try:
                    parsed = json.loads(raw)
                    tool_result = json.dumps(parsed, indent=2)
                except Exception:
                    tool_result = raw

                print("--- Tool Result ---")
                print(tool_result)

                # Append tool result to history
                self.messages.append({"role": "tool_result", "content": tool_result})
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=f"Tool result: {tool_result}")],
                    )
                )

                # Final call without tools
                final_response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=0),
                )
                text = final_response.text
            else:
                # No function call, use direct text
                text = response.text

            # Save assistant response
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

    async def connect_to_server_and_run(self):
        try:
            server_params = StdioServerParameters(
                command="uv",
                args=["run", "server.py"],
                env=None,
            )
            print("Connecting to MCP server...")
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session
                    await session.initialize()
                    response = await session.list_tools()
                    tools = response.tools
                    print("\nConnected to server with tools:", [t.name for t in tools])
                    self.available_tools = [
                        {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
                        for t in tools
                    ]
                    await self.chat_loop()
        except Exception as e:
            print(f"Error connecting to server: {e}")
            traceback.print_exc()


async def main():
    try:
        print("Starting MCP Chatbot...")
        chatbot = MCP_ChatBot()
        await chatbot.connect_to_server_and_run()
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
