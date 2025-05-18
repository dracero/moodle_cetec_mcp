from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import List, Dict, Any
import asyncio
import nest_asyncio
import os
import json
import traceback

nest_asyncio.apply()

load_dotenv()

class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.session: ClientSession = None
        
        # Check if API key exists
        if not os.environ.get("GOOGLE_API_KEY"):
            print("ERROR: GOOGLE_API_KEY not found in environment variables!")
            print("Please add your Google API key to .env file")
        
        # Initialize the Gemini model through LangChain
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=os.environ.get("GOOGLE_API_KEY"),
                temperature=0.2
            )
            print("Gemini model initialized successfully")
        except Exception as e:
            print(f"Error initializing Gemini: {str(e)}")
            traceback.print_exc()
            raise e
            
        self.available_tools: List[dict] = []
        self.messages = []

    async def process_query(self, query):
        # Add the user query to the message history
        self.messages.append(HumanMessage(content=query))
        
        try:
            # Format tools for Gemini
            tools_for_gemini = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"]
                    }
                } for tool in self.available_tools
            ]
            
            # Print debug info
            print(f"Sending query with {len(tools_for_gemini)} tools to Gemini")
            
            # Get response from Gemini
            try:
                # First try without tools for debugging
                simple_response = self.llm.invoke(self.messages)
                print(f"Simple response received: {simple_response.content[:100]}...")
                
                # Now try with tools
                response = self.llm.invoke(
                    self.messages,
                    tools=tools_for_gemini
                )
                print("Response received from Gemini")
                
                # Correctly add the AI response to message history as AIMessage
                # This is the key fix - we need to properly format the response as an AIMessage
                self.messages.append(AIMessage(content=response.content))
                
                # Print response to user
                print(response.content)
                
            except Exception as e:
                print(f"Error invoking Gemini: {str(e)}")
                traceback.print_exc()
                return
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            traceback.print_exc()

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot with Gemini Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                    
                await self.process_query(query)
                print("\n")
                    
            except Exception as e:
                print(f"\nError in chat loop: {str(e)}")
                traceback.print_exc()
    
    async def connect_to_server_and_run(self):
        try:
            # Create server parameters for stdio connection
            server_params = StdioServerParameters(
                command="uv",  # Executable
                args=["run", "server.py"],  # Optional command line arguments
                env=None,  # Optional environment variables
            )
            
            print("Connecting to MCP server...")
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session
                    # Initialize the connection
                    await session.initialize()
        
                    # List available tools
                    response = await session.list_tools()
                    
                    tools = response.tools
                    print("\nConnected to server with tools:", [tool.name for tool in tools])
                    
                    self.available_tools = [{
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    } for tool in response.tools]
        
                    await self.chat_loop()
        except Exception as e:
            print(f"Error connecting to server: {str(e)}")
            traceback.print_exc()


async def main():
    try:
        print("Starting MCP Chatbot...")
        chatbot = MCP_ChatBot()
        await chatbot.connect_to_server_and_run()
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())