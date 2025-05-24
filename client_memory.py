from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import List, Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from google import genai
from google.genai import types
import asyncio
import nest_asyncio
import os
import json
import traceback
import uuid
from contextlib import AsyncExitStack
from datetime import datetime
import sqlite3

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


class SimpleMemory:
    """Simple conversation memory without embeddings"""
    
    def __init__(self, db_path="conversation_memory.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing conversations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_message TEXT,
                ai_response TEXT,
                session_id TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_conversation(self, user_message: str, ai_response: str, session_id: str):
        """Store conversation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (timestamp, user_message, ai_response, session_id)
            VALUES (?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            user_message,
            ai_response,
            session_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_conversations(self, session_id: str, limit: int = 3):
        """Get recent conversations from this session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_message, ai_response, timestamp
            FROM conversations 
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (session_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return results


class AgentState(MessagesState):
    """Extended state that inherits from MessagesState for proper LangGraph integration"""
    tool_call: Optional[Dict] = None
    tool_call_id: Optional[str] = None
    session_id: str = ""


class MCP_ChatBot:
    def __init__(self):
        self.sessions: List[ClientSession] = []
        self.tool_to_session = {}
        self.exit_stack = AsyncExitStack()
        self.model_name = config.get("model", "gemini-2.0-flash")
        self.session_id = str(uuid.uuid4())  # Unique session ID
        
        # Initialize simple memory
        self.memory = SimpleMemory()
        
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
        
        self.available_tools: List[Dict] = []

    def clean_schema(self, schema: Dict) -> Dict:
        """Clean schema to remove properties that cause validation errors in Gemini."""
        if not isinstance(schema, dict):
            return schema
        
        cleaned = {}
        for key, value in schema.items():
            if key in ["additionalProperties", "$schema"]:
                continue
            if isinstance(value, dict):
                if key == "properties":
                    cleaned[key] = {prop_key: self.clean_schema(prop_value) 
                                  for prop_key, prop_value in value.items()}
                else:
                    cleaned[key] = self.clean_schema(value)
            elif isinstance(value, list):
                cleaned[key] = [self.clean_schema(item) if isinstance(item, dict) else item 
                              for item in value]
            else:
                cleaned[key] = value
        return cleaned

    async def connect_to_server(self, server_name: str, server_config: Dict) -> None:
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            
            self.sessions.append(session)
            response = await session.list_tools()
            tools = response.tools
            
            print(f"Connected to {server_name} with tools:", [t.name for t in tools])
            
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
        print("=" * 50)
        print("DETAILED SERVER RESPONSE")
        print("=" * 50)
        print(f"Result type: {type(result)}")
        
        if hasattr(result, 'content') and result.content:
            print(f"Content count: {len(result.content)}")
            for i, content_item in enumerate(result.content):
                print(f"Content {i+1}:")
                print(f"  Type: {type(content_item)}")
                if hasattr(content_item, 'text'):
                    print(f"  Text: {content_item.text}")
        
        if hasattr(result, 'isError') and result.isError:
            print(f"Error detected: {result.isError}")
        
        print("=" * 50)

    async def call_model(self, state: AgentState):
        """Model calling with memory integration - similar to original"""
        messages = state["messages"]
        
        # Get recent context from memory
        recent_convs = self.memory.get_recent_conversations(self.session_id, limit=2)
        
        contents = []
        
        # Add recent context if available
        if recent_convs:
            context_parts = []
            for user_msg, ai_resp, timestamp in recent_convs:
                context_parts.append(f"Previous: User: {user_msg} | AI: {ai_resp}")
            
            context = "\n".join(context_parts)
            contents.append(types.Content(
                role="user", 
                parts=[types.Part.from_text(text=f"Context from recent conversation:\n{context}\n\n")]
            ))
        
        # Add current conversation messages
        for msg in messages:
            if isinstance(msg, HumanMessage):
                contents.append(types.Content(
                    role="user", 
                    parts=[types.Part.from_text(text=msg.content)]
                ))
            elif isinstance(msg, AIMessage):
                contents.append(types.Content(
                    role="model", 
                    parts=[types.Part.from_text(text=msg.content)]
                ))
            elif isinstance(msg, ToolMessage):
                contents.append(types.Content(
                    role="user", 
                    parts=[types.Part.from_text(text=f"Tool result: {msg.content}")]
                ))
        
        # Prepare Gemini tools
        gemini_tools = []
        for tool in self.available_tools:
            gemini_tool = types.Tool(
                function_declarations=[{
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                }]
            )
            gemini_tools.append(gemini_tool)
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                tools=gemini_tools,
            ),
        )
        
        candidate = response.candidates[0]
        
        # Check for function calls (like in original)
        if (candidate.content.parts and 
            getattr(candidate.content.parts[0], 'function_call', None)):
            function_call = candidate.content.parts[0].function_call
            tool_call_id = str(uuid.uuid4())
            
            return {
                "tool_call": {
                    "name": function_call.name,
                    "args": dict(function_call.args)
                },
                "tool_call_id": tool_call_id
            }
        else:
            ai_response = response.text
            
            # Store conversation in memory only for final responses
            user_message = ""
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    user_message = msg.content
                    break
            
            if user_message and not any(isinstance(msg, ToolMessage) for msg in messages):
                # Only store if this is a complete conversation (not mid-tool-use)
                self.memory.store_conversation(user_message, ai_response, self.session_id)
            
            return {
                "messages": [AIMessage(content=ai_response)]
            }

    async def execute_tool(self, state: AgentState):
        """Execute tool - exactly like original but with better error handling"""
        tool_call = state["tool_call"]
        tool_call_id = state["tool_call_id"]
        
        if not tool_call or not tool_call_id:
            return state
        
        name = tool_call["name"]
        args = tool_call["args"]
        
        try:
            session = self.tool_to_session.get(name, self.sessions[0])
            result = await session.call_tool(name, arguments=args)
            
            self.print_server_response_details(result)
            
            # Extract content from result (same as original)
            if result.content:
                all_content = []
                for content_item in result.content:
                    if hasattr(content_item, 'text') and content_item.text:
                        all_content.append(content_item.text)
                    elif hasattr(content_item, 'data'):
                        all_content.append(str(content_item.data))
                    else:
                        all_content.append(str(content_item))
                raw_response = ''.join(all_content) if all_content else str(result)
            else:
                raw_response = str(result)
            
            # Try to format as JSON (same as original)
            try:
                parsed = json.loads(raw_response)
                tool_result = json.dumps(parsed, indent=2)
            except Exception:
                tool_result = raw_response
            
            return {
                "messages": [ToolMessage(content=tool_result, tool_call_id=tool_call_id)],
                "tool_call": None,
                "tool_call_id": None
            }
            
        except Exception as e:
            error_message = f"Error executing tool {name}: {str(e)}"
            print(f"❌ {error_message}")
            
            return {
                "messages": [ToolMessage(content=error_message, tool_call_id=tool_call_id)],
                "tool_call": None,
                "tool_call_id": None
            }

    def should_continue(self, state: AgentState):
        """Determine next step - exactly like original"""
        if state.get("tool_call"):
            return "action"
        return END

    async def run_langgraph_agent(self):
        """LangGraph agent workflow - restored to original structure with memory"""
        
        # Define the workflow - same structure as original
        workflow = StateGraph(AgentState)
        workflow.add_node("model", self.call_model)
        workflow.add_node("action", self.execute_tool)
        workflow.add_edge(START, "model")
        workflow.add_conditional_edges("model", self.should_continue)
        workflow.add_edge("action", "model")

        # Initialize checkpointer with MemorySaver - corrected implementation
        memory_saver = MemorySaver()
        graph = workflow.compile(checkpointer=memory_saver)
        
        print("🚀 LangGraph MCP Chatbot with Memory Started!")
        print(f"📝 Session ID: {self.session_id}")
        print("💬 Type your queries or 'quit' to exit.")
        print("🧠 Memory active - I'll remember our conversations!")
        
        while True:
            try:
                query = input("\n🔍 Query: ").strip()
                if query.lower() in ['quit', 'exit']:
                    break
                
                # Configuration for thread - proper LangGraph thread handling
                config = {"configurable": {"thread_id": self.session_id}}
                
                # Run the workflow - same as original but with proper config
                result = await graph.ainvoke(
                    {"messages": [HumanMessage(content=query)], "session_id": self.session_id}, 
                    config=config
                )
                
                print("\n🎯 AI FINAL RESPONSE " + "=" * 30)
                
                # Get the final AI response - same as original
                if result["messages"]:
                    final_message = result["messages"][-1]
                    if isinstance(final_message, AIMessage):
                        print(final_message.content)
                    else:
                        print("Tool execution completed.")
                else:
                    print("No response generated.")
                
                print("=" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break  
            except Exception as e:
                print(f"❌ Error in chat loop: {e}")
                traceback.print_exc()
        
        await self.exit_stack.aclose()


async def main():
    try:
        print("🔄 Starting MCP Chatbot...")
        chatbot = MCP_ChatBot()
        await chatbot.connect_to_servers()
        await chatbot.run_langgraph_agent()
    except Exception as e:
        print(f"❌ Error in main: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())