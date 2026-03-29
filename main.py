"""
J.A.R.V.I.S — Main Entry Point
Just A Rather Very Intelligent System
"""

import sys
import os
import asyncio
import signal
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from core.jarvis import JARVIS
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint
import colorama

colorama.init()
console = Console()

BANNER = """
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝

   Just A Rather Very Intelligent System v1.0
   Autonomous • Self-Evolving • Controlled
"""


def display_banner():
    console.print(Panel(
        Text(BANNER, style="bold cyan", justify="center"),
        border_style="bright_blue",
        padding=(1, 4)
    ))
    console.print(
        "[dim cyan]Initializing cognitive architecture...[/dim cyan]\n"
    )


def display_help():
    help_text = """
[bold cyan]J.A.R.V.I.S Command Reference[/bold cyan]

[yellow]Conversational Commands:[/yellow]
  [green]<any text>[/green]          → Natural language input (J.A.R.V.I.S interprets automatically)
  [green]help[/green]                → Show this help menu
  [green]status[/green]              → System status and performance metrics
  [green]memory[/green]              → View memory contents
  [green]history[/green]             → View last 10 executed tasks
  [green]strategies[/green]          → List learned strategies

[yellow]System Commands:[/yellow]
  [green]mode engineer[/green]       → Switch to detailed technical mode
  [green]mode execution[/green]      → Switch to direct action mode
  [green]mode adaptive[/green]       → Switch to adaptive mode (default)
  [green]evolve[/green]              → Trigger manual evolution cycle
  [green]diagnose[/green]            → Run self-diagnostic
  [green]benchmark[/green]           → Run performance benchmark
  [green]dashboard[/green]           → Launch web dashboard

[yellow]Control:[/yellow]
  [green]STOP[/green] / [green]quit[/green] / [green]exit[/green]   → Shutdown J.A.R.V.I.S safely
  [green]Ctrl+C[/green]              → Emergency stop

[dim]All actions are logged and pass safety validation automatically.[/dim]
"""
    console.print(Panel(help_text, title="[bold blue]⚡ Command Reference[/bold blue]", border_style="blue"))


async def run_interactive(jarvis: JARVIS):
    """Run the interactive CLI loop."""
    display_help()

    console.print("[bold green]✓ J.A.R.V.I.S is online. All systems nominal.[/bold green]\n")

    while True:
        try:
            # Prompt
            user_input = console.input("[bold cyan]YOU →[/bold cyan] ").strip()

            if not user_input:
                continue

            # Emergency Stop
            if user_input.upper() in {"STOP", "QUIT", "EXIT"}:
                console.print("\n[bold yellow]J.A.R.V.I.S → Initiating graceful shutdown...[/bold yellow]")
                await jarvis.shutdown()
                console.print("[bold green]J.A.R.V.I.S → All systems offline. Standing by.[/bold green]")
                break

            # Built-in commands
            if user_input.lower() == "help":
                display_help()
                continue

            if user_input.lower() == "dashboard":
                console.print("[cyan]J.A.R.V.I.S → Launching web dashboard...[/cyan]")
                await jarvis.launch_dashboard()
                continue

            # Process through J.A.R.V.I.S
            response = await jarvis.process(user_input)

            if response:
                console.print(
                    Panel(
                        response,
                        title="[bold blue]⚡ J.A.R.V.I.S[/bold blue]",
                        border_style="bright_blue",
                        padding=(0, 2)
                    )
                )

        except KeyboardInterrupt:
            console.print("\n[bold red]⚡ Emergency stop triggered. Shutting down...[/bold red]")
            await jarvis.emergency_stop()
            break
        except EOFError:
            break
        except Exception as e:
            console.print(f"[bold red]System Error: {e}[/bold red]")


def handle_signal(signum, frame):
    console.print("\n[bold red]Signal received. Initiating emergency shutdown...[/bold red]")
    sys.exit(0)


async def main():
    display_banner()

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Initialize J.A.R.V.I.S
    console.print("[dim]Loading cognitive modules...[/dim]")
    jarvis = JARVIS()

    with console.status("[bold cyan]Booting J.A.R.V.I.S...[/bold cyan]", spinner="dots"):
        await jarvis.initialize()

    console.print("[bold bright_green]✓ All systems initialized.[/bold bright_green]")
    console.print(f"[dim]Memory loaded | RL engine ready | Safety governor armed[/dim]\n")

    await run_interactive(jarvis)


if __name__ == "__main__":
    asyncio.run(main())
