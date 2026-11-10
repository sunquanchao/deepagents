import argparse
import os
import sys

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Load environment variables
load_dotenv()

console = Console()


def create_sql_deep_agent():
    """Create and return a text-to-SQL Deep Agent"""

    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Connect to Chinook database
    db_path = os.path.join(base_dir, "chinook.db")
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)

    # Initialize the model.
    # Modified: read the model name from the ANTHROPIC_MODEL env var so that
    # Anthropic-compatible endpoints (e.g. BigModel/GLM) can be used without
    # code changes. The original hardcoded Claude configuration is kept below
    # for reference; restore it if you have direct Anthropic API access.
    #
    # Original configuration (requires an Anthropic API key):
    # model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    model = ChatAnthropic(model=model_name, temperature=0)

    # Create SQL toolkit and get tools
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    sql_tools = toolkit.get_tools()

    # Create the Deep Agent with all parameters
    agent = create_deep_agent(
        model=model,  # Model from ANTHROPIC_MODEL (default: Claude Sonnet 4.5), temperature=0
        memory=["./AGENTS.md"],  # Agent identity and general instructions
        skills=[
            "./skills/"
        ],  # Specialized workflows (query-writing, schema-exploration)
        tools=sql_tools,  # SQL database tools
        subagents=[],  # No subagents needed
        # Persistent file storage; virtual_mode=False keeps real-filesystem
        # path semantics (pinned explicitly: the default flips in 0.6.0)
        backend=FilesystemBackend(root_dir=base_dir, virtual_mode=False),
    )

    return agent


def echo_sql_calls(message) -> None:
    """Echo SQL emitted by SQL toolkit tool calls to the user"""
    for call in getattr(message, "tool_calls", None) or []:
        query = (call.get("args") or {}).get("query")
        if query:
            # Rendered as plain Text so rich markup in the SQL is not interpreted
            console.print(
                Text(f"[{call.get('name', 'tool')}] ", style="bold cyan"),
                Text(query, style="cyan"),
            )


def main():
    """Main entry point for the SQL Deep Agent CLI"""
    parser = argparse.ArgumentParser(
        description="Text-to-SQL Deep Agent powered by LangChain Deep Agents and Claude Sonnet 4.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py "What are the top 5 best-selling artists?"
  python agent.py "Which employee generated the most revenue by country?"
  python agent.py "How many customers are from Canada?"
        """,
    )
    parser.add_argument(
        "question",
        type=str,
        help="Natural language question to answer using the Chinook database",
    )

    args = parser.parse_args()

    # Display the question
    console.print(
        Panel(f"[bold cyan]Question:[/bold cyan] {args.question}", border_style="cyan")
    )
    console.print()

    # Create the agent
    console.print("[dim]Creating SQL Deep Agent...[/dim]")
    agent = create_sql_deep_agent()

    # Invoke the agent
    console.print("[dim]Processing query...[/dim]\n")

    try:
        # Stream intermediate steps so executed SQL is echoed while it runs
        final_message = None
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": args.question}]},
            stream_mode="updates",
        ):
            for update in chunk.values():
                if not isinstance(update, dict):
                    continue
                for message in update.get("messages", []):
                    final_message = message
                    echo_sql_calls(message)

        # Extract and display the final answer
        answer = (
            final_message.content
            if hasattr(final_message, "content")
            else str(final_message)
        )
        # Reasoning models return a list of content blocks (thinking, text, ...);
        # display thinking separately from the final answer
        thinking = ""
        if isinstance(answer, list):
            thinking = "\n".join(
                block.get("thinking", "")
                for block in answer
                if isinstance(block, dict) and block.get("type") == "thinking"
            )
            answer = "\n".join(
                block.get("text", "")
                for block in answer
                if isinstance(block, dict) and block.get("type") == "text"
            )

        if thinking:
            # Rendered as plain Text so rich markup in the reasoning is not interpreted
            console.print(
                Panel(
                    Text(thinking, style="dim"),
                    title="Thinking",
                    border_style="dim",
                )
            )
            console.print()

        console.print(
            Panel(f"[bold green]Answer:[/bold green]\n\n{answer}", border_style="green")
        )

    except Exception as e:
        console.print(
            Panel(f"[bold red]Error:[/bold red]\n\n{str(e)}", border_style="red")
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
