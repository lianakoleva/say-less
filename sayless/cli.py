"""Command-line interface for SayLess."""

import asyncio
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import click
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import get_settings
from .core import Assistant
from .queue import QueueManager
from .data.contacts import ContactsManager
from .data.notes import NotesManager
from .data.chat import ChatManager
from .tools import model_tools

console = Console()


@click.group()
@click.version_option(version="0.1.0")
@click.option('--quiet', is_flag=True, envvar='SAYLESS_QUIET', help='Suppress verbose output')
@click.pass_context
def main(ctx, quiet):
    """SayLess - Local voice assistant with STT and LLM."""
    ctx.ensure_object(dict)
    ctx.obj['quiet'] = quiet
    if quiet:
        import os
        os.environ['SAYLESS_QUIET'] = '1'


def _server_root(base_url: str) -> str:
    root = base_url.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def _llm_server_up(base_url: str) -> bool:
    """Return True if the LiteRT-LM server responds."""
    try:
        response = httpx.get(f"{_server_root(base_url)}/health", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def _server_health(base_url: str) -> dict:
    """Return the server's /health payload, or an empty dict."""
    try:
        response = httpx.get(f"{_server_root(base_url)}/health", timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _show_log_tail(log_path: Path, lines: int = 30) -> None:
    """Print the end of the LLM server log, if it exists."""
    try:
        text = log_path.read_text(errors="replace").splitlines()[-lines:]
    except Exception:
        return
    if text:
        console.print("\n".join(text), markup=False, style="dim")


def _start_llm_server() -> bool:
    """Start the LiteRT-LM server and wait for it to accept requests."""
    settings = get_settings()
    if _llm_server_up(settings.llm_base_url):
        return True

    model_path = Path(settings.llm_model_path).expanduser()
    if not model_path.is_file():
        console.print(f"[red]LiteRT-LM model not found:[/red] {model_path}")
        console.print(
            "Set [cyan]SAYLESS_LLM_MODEL_PATH[/cyan] to a "
            "mobile_actions_q8_ekv1024.litertlm file."
        )
        return False

    console.print("[cyan]Starting LiteRT-LM server...[/cyan]")
    log_path = settings.data_dir / "logs" / "llm.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "sayless.llm.server",
        "--model",
        str(model_path),
        "--host",
        settings.llm_server_host,
        "--port",
        str(settings.llm_server_port),
    ]
    with open(log_path, "ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )

    console.print(
        f"[dim]Loading the model (first run can take a minute). "
        f"Log: {log_path}[/dim]"
    )

    deadline = time.time() + 300
    while time.time() < deadline:
        if _llm_server_up(settings.llm_base_url):
            return True
        if process.poll() is not None:
            console.print("[red]LiteRT-LM server exited during startup.[/red]")
            _show_log_tail(log_path)
            return False
        time.sleep(0.5)

    console.print("[red]LiteRT-LM server did not respond within 5 minutes.[/red]")
    console.print(f"[dim]Server log: {log_path}[/dim]")
    _show_log_tail(log_path)
    return False


def _ensure_llm_server() -> bool:
    """Make sure the LiteRT-LM server is running when the LLM is enabled."""
    settings = get_settings()
    if not settings.llm_enabled:
        return True

    if _llm_server_up(settings.llm_base_url):
        return True

    if _start_llm_server():
        return True

    console.print("[red]Could not start the LiteRT-LM server.[/red]")
    console.print(
        f"Start it manually: [cyan]python -m sayless.llm.server "
        f"--model {settings.llm_model_path}[/cyan]"
    )
    console.print("Then rerun the command, or run [cyan]sayless setup[/cyan].")
    return False


@main.command()
@click.option("--model-path", default=None, help="Path to the .litertlm model")
@click.option("--host", default=None, help="LiteRT-LM server host")
@click.option("--port", type=int, default=None, help="LiteRT-LM server port")
@click.option("--start/--no-start", default=True, help="Start the server after setup")
@click.option("--yes", is_flag=True, help="Answer yes to setup prompts")
def setup(model_path: Optional[str], host: Optional[str], port: Optional[int], start: bool, yes: bool):
    """Verify the LiteRT-LM model and start the local LLM server."""
    settings = get_settings()
    path = Path(model_path or settings.llm_model_path).expanduser()
    host = host or settings.llm_server_host
    port = port or settings.llm_server_port

    console.print(Panel.fit("[bold green]SayLess Setup[/bold green]", border_style="green"))
    console.print(f"Model: [cyan]{path}[/cyan]")
    console.print(f"Server: [cyan]http://{host}:{port}/v1[/cyan]\n")

    if importlib.util.find_spec("litert_lm") is None:
        console.print("[red]litert-lm is not installed.[/red]")
        console.print("Install it with: [cyan]pip install 'litert-lm>=0.18'[/cyan]")
        raise SystemExit(1)

    if not path.is_file():
        console.print(f"[red]Model file not found:[/red] {path}")
        console.print(
            "\nCopy the Gemma LiteRT-LM file to this machine, for example:\n"
            "  [cyan]mobile_actions_q8_ekv1024.litertlm[/cyan]\n"
            "Then run:\n"
            "  [cyan]sayless setup --model-path /path/to/mobile_actions_q8_ekv1024.litertlm[/cyan]\n"
            "or set [cyan]SAYLESS_LLM_MODEL_PATH[/cyan] in .env."
        )
        raise SystemExit(1)

    if start and not _llm_server_up(settings.llm_base_url):
        # Use the values from this invocation, not only the .env values.
        settings.llm_model_path = path
        settings.llm_server_host = host
        settings.llm_server_port = port
        settings.llm_base_url = f"http://{host}:{port}/v1"
        if not _start_llm_server():
            console.print("[red]LiteRT-LM server did not start.[/red]")
            raise SystemExit(1)

    if not _llm_server_up(settings.llm_base_url):
        console.print("[yellow]Server is not running.[/yellow] Start it with:")
        console.print(
            f"  [cyan]python -m sayless.llm.server --model {path} "
            f"--host {host} --port {port}[/cyan]"
        )
        raise SystemExit(1)

    console.print("[green]LiteRT-LM server is ready.[/green]")
    console.print(
        "\nAdd this to [cyan].env[/cyan] if it is not already there:\n"
        f"  SAYLESS_LLM_MODEL_PATH={path}\n"
        f"  SAYLESS_LLM_BASE_URL=http://{host}:{port}/v1\n"
    )
    console.print("[green]Setup complete![/green] Run 'sayless run' or 'sayless chat'.")


@main.command()
@click.option("--model-path", default=None, help="Path to the .litertlm model")
@click.option("--host", default=None, help="Bind host")
@click.option("--port", type=int, default=None, help="Bind port")
def serve(model_path: Optional[str], host: Optional[str], port: Optional[int]):
    """Run the LiteRT-LM server in the foreground."""
    settings = get_settings()
    command = [
        sys.executable,
        "-m",
        "sayless.llm.server",
        "--model",
        str(Path(model_path or settings.llm_model_path).expanduser()),
        "--host",
        host or settings.llm_server_host,
        "--port",
        str(port or settings.llm_server_port),
    ]
    os.execv(sys.executable, command)


@main.command()
def doctor():
    """Check that SayLess dependencies and the LiteRT-LM server are ready."""
    settings = get_settings()

    table = Table(title="SayLess Doctor")
    table.add_column("Check", style="cyan")
    table.add_column("Status")

    def add(label: str, ok: bool, detail: str = ""):
        status = "[green]OK[/green]" if ok else "[red]MISSING[/red]"
        if detail:
            status = f"{status} [dim]{detail}[/dim]"
        table.add_row(label, status)

    model_path = Path(settings.llm_model_path).expanduser()
    litert_ok = importlib.util.find_spec("litert_lm") is not None
    add("LiteRT-LM installed", litert_ok,
        "" if litert_ok else "pip install 'litert-lm>=0.18'")
    add("LLM model file", model_path.is_file(), str(model_path))
    server_up = _llm_server_up(settings.llm_base_url)
    add("LiteRT-LM server", server_up, settings.llm_base_url)
    if server_up:
        health = _server_health(settings.llm_base_url)
        if "supports_tools" in health:
            native = bool(health["supports_tools"])
            add(
                "Function calling",
                True,
                "native FunctionGemma tools" if native else "native first, JSON fallback",
            )

    for module, label in [
        ("faster_whisper", "faster-whisper"),
        ("sounddevice", "sounddevice"),
        ("soundfile", "soundfile"),
    ]:
        present = importlib.util.find_spec(module) is not None
        add(label, present, "" if present else f"pip install {module}")

    arecord = shutil.which("arecord")
    add("arecord (CoralBoard mic)", arecord is not None,
        "" if arecord else "alsa-utils")
    add("STT model", True, settings.stt_model)
    if settings.google_maps_api_key:
        table.add_row(
            "Google Maps key",
            "[green]SET[/green] [dim]requires the legacy Directions API[/dim]",
        )
    else:
        table.add_row(
            "Google Maps key",
            "[yellow]NOT SET[/yellow] [dim]using free OSRM routing[/dim]",
        )
    add("Queue directory", settings.queue_dir.exists(), str(settings.queue_dir))
    console.print(table)

    if not (server_up and model_path.is_file()):
        console.print("\n[cyan]Run 'sayless setup' to start the LiteRT-LM server.[/cyan]")


@main.command()
def tools():
    """Show what SayLess can do in plain language."""
    console.print(model_tools.help_text())


@main.command()
@click.option('--stt/--no-stt', default=None, help='Enable/disable STT')
@click.option('--llm/--no-llm', default=None, help='Enable/disable LLM')
def run(stt, llm):
    """Run the voice assistant."""
    settings = get_settings()
    if stt is not None:
        settings.stt_enabled = stt
    if llm is not None:
        settings.llm_enabled = llm

    async def _run() -> bool:
        if not _ensure_llm_server():
            raise SystemExit(1)
        assistant = Assistant()
        if not await assistant.initialize():
            await assistant.cleanup()
            console.print("[bold red]Initialization failed. Exiting.[/bold red]")
            return False

        console.print(Panel.fit(
            "[bold green]SayLess[/bold green] is running!\n"
            "Say 'exit', 'quit', or press Ctrl+C to stop.",
            border_style="green"
        ))

        try:
            while True:
                console.print("\n[cyan]Press Enter to record (Enter again stops), or type 'text:' followed by message[/cyan]")
                user_input = input().strip()

                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                elif user_input.lower().startswith('text:'):
                    text = user_input[5:].strip()
                    if text:
                        await assistant.process_text(text)
                elif user_input == '':
                    # Voice input: the Enter that got us here starts the recording.
                    await assistant.process_voice(start_prompt=False)
                else:
                    # Treat as text
                    await assistant.process_text(user_input)

                # Process queue periodically
                await assistant.process_queue()

        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
        finally:
            await assistant.cleanup()
        return True

    if not asyncio.run(_run()):
        raise SystemExit(1)


@main.command()
def chat():
    """Start text chat mode."""
    async def _chat() -> bool:
        if not _ensure_llm_server():
            raise SystemExit(1)
        assistant = Assistant()
        if not await assistant.initialize():
            await assistant.cleanup()
            console.print("[bold red]Initialization failed. Exiting.[/bold red]")
            return False

        console.print(Panel.fit(
            "[bold green]SayLess Chat Mode[/bold green]\n"
            "Type 'exit' or 'quit' to stop.",
            border_style="green"
        ))

        try:
            while True:
                user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
                if user_input.lower() in ['exit', 'quit']:
                    break
                if not user_input:
                    continue
                await assistant.process_text(user_input)
                await assistant.process_queue()
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
        finally:
            await assistant.cleanup()
        return True

    if not asyncio.run(_chat()):
        raise SystemExit(1)


@main.command()
def listen():
    """Listen once and transcribe."""
    async def _listen() -> bool:
        if not _ensure_llm_server():
            raise SystemExit(1)
        assistant = Assistant()
        if not await assistant.initialize():
            await assistant.cleanup()
            console.print("[bold red]Initialization failed. Exiting.[/bold red]")
            return False
        await assistant.process_voice()
        await assistant.cleanup()
        return True

    if not asyncio.run(_listen()):
        raise SystemExit(1)


@main.command()
def config():
    """Show current configuration."""
    settings = get_settings()
    
    table = Table(title="SayLess Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Data Dir", str(settings.data_dir))
    table.add_row("Queue Dir", str(settings.queue_dir))
    table.add_row("STT Enabled", str(settings.stt_enabled))
    table.add_row("STT Engine", settings.stt_engine)
    table.add_row("STT Model", settings.stt_model)
    table.add_row("STT Audio Device", settings.stt_audio_device)
    table.add_row("LLM Enabled", str(settings.llm_enabled))
    table.add_row("LLM Engine", settings.llm_engine)
    table.add_row("LLM Model", settings.llm_model_name)
    table.add_row("LLM Model Path", str(settings.llm_model_path))
    table.add_row("LLM Base URL", settings.llm_base_url)
    table.add_row("Phone Tools", str(settings.enable_phone_tools))
    table.add_row("Phone Provider", "command" if (settings.phone_call_command or settings.phone_text_command) else "simulated log")
    table.add_row("Maps Tools", str(settings.enable_maps_tools))
    table.add_row("Navigation Provider", "Google Directions (OSRM fallback)" if settings.google_maps_api_key else "OSRM (free fallback)")
    table.add_row("Google Maps Key", "set" if settings.google_maps_api_key else "not set")
    table.add_row("Current Location", settings.current_location or "ask each route")
    table.add_row("Navigation Units", settings.navigation_units)
    table.add_row("Weather Tools", str(settings.enable_weather_tools))
    table.add_row("Weather Location", settings.weather_location or "ask on first use")
    table.add_row("Seed Contacts", str(settings.seed_contacts))
    table.add_row("Queue Enabled", str(settings.queue_enabled))
    
    console.print(table)
    console.print(f"\n[dim]Config file: .env (or env vars with SAYLESS_ prefix)[/dim]")


@main.command()
def profile():
    """Show notes SayLess has saved about you."""
    async def _profile():
        from .data.profile import UserProfile

        data = await UserProfile().get_all()
        if not data:
            console.print("[yellow]No saved preferences yet[/yellow]")
            return
        table = Table(title="User Profile")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        for key, value in sorted(data.items()):
            table.add_row(str(key), str(value))
        console.print(table)
        console.print(f"[dim]{get_settings().data_dir / 'profile'}[/dim]")

    asyncio.run(_profile())


@main.command()
def queue():
    """Show queued tasks."""
    async def _queue():
        qm = QueueManager()
        tasks = await qm.get_all()
        
        if not tasks:
            console.print("[yellow]No tasks in queue[/yellow]")
            return
        
        table = Table(title="Task Queue")
        table.add_column("ID", style="cyan", max_width=8)
        table.add_column("Tool", style="magenta")
        table.add_column("Action", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Created", style="dim")
        
        for task in tasks:
            table.add_row(
                task['id'][:8],
                task['tool_name'],
                task['action'],
                task['status'],
                task['created_at'][:19]
            )
        
        console.print(table)
        console.print(f"\n[dim]Queue location: {qm.get_queue_path()}[/dim]")
    
    asyncio.run(_queue())


@main.command(name='queue-clear')
@click.option('--clear-completed', is_flag=True, help='Clear completed tasks')
@click.option('--clear-all', is_flag=True, help='Clear all tasks')
def queue_clear_cmd(clear_completed, clear_all):
    """Clear queued tasks."""
    async def _clear():
        qm = QueueManager()
        if clear_all:
            await qm.clear_all()
            console.print("[green]All tasks cleared[/green]")
        elif clear_completed:
            await qm.clear_completed()
            console.print("[green]Completed tasks cleared[/green]")
        else:
            console.print("[yellow]Use --clear-completed or --clear-all[/yellow]")
    
    asyncio.run(_clear())


@main.command(name='process-queue')
def process_queue_cmd():
    """Process queued tasks if internet is available."""
    async def _process() -> bool:
        if not _ensure_llm_server():
            raise SystemExit(1)
        assistant = Assistant()
        if not await assistant.initialize():
            await assistant.cleanup()
            console.print("[bold red]Initialization failed. Exiting.[/bold red]")
            return False
        console.print("[cyan]Processing queue...[/cyan]")
        await assistant.process_queue()
        console.print("[green]Done[/green]")
        await assistant.cleanup()
        return True

    if not asyncio.run(_process()):
        raise SystemExit(1)


@main.group()
def contacts():
    """Manage contacts."""
    pass


@contacts.command(name='add')
@click.argument('name')
@click.option('--phone', help='Phone number')
@click.option('--email', help='Email address')
def contacts_add(name, phone, email):
    """Add a contact."""
    async def _add():
        cm = ContactsManager()
        contact_id = await cm.add_contact(name, phone=phone, email=email)
        console.print(f"[green]Added contact: {name} ({contact_id})[/green]")
    
    asyncio.run(_add())


@contacts.command(name='list')
def contacts_list():
    """List all contacts."""
    async def _list():
        settings = get_settings()
        cm = ContactsManager()
        if settings.seed_contacts:
            await cm.seed_demo_contacts()
        contacts_list = await cm.list_contacts()
        if not contacts_list:
            console.print("[yellow]No contacts[/yellow]")
            return
        table = Table(title="Contacts")
        table.add_column("Name", style="cyan")
        table.add_column("Phone", style="green")
        table.add_column("Email", style="blue")
        for c in contacts_list:
            table.add_row(c.get('name', ''), c.get('phone', '') or '', c.get('email', '') or '')
        console.print(table)
    
    asyncio.run(_list())


@contacts.command(name='seed')
def contacts_seed():
    """Add any missing demo contacts."""
    async def _seed():
        cm = ContactsManager()
        added = await cm.seed_demo_contacts()
        if added:
            console.print(f"[green]Seeded {added} demo contacts[/green]")
        else:
            console.print("[yellow]All demo contacts already exist[/yellow]")

    asyncio.run(_seed())


@contacts.command(name='search')
@click.argument('query')
def contacts_search(query):
    """Search contacts."""
    async def _search():
        cm = ContactsManager()
        results = await cm.search_contacts(query)
        if not results:
            console.print("[yellow]No contacts found[/yellow]")
            return
        table = Table(title=f"Search results for '{query}'")
        table.add_column("Name", style="cyan")
        table.add_column("Phone", style="green")
        table.add_column("Email", style="blue")
        for c in results:
            table.add_row(c.get('name', ''), c.get('phone', '') or '', c.get('email', '') or '')
        console.print(table)
    
    asyncio.run(_search())


@main.group()
def notes():
    """Manage notes."""
    pass


@notes.command(name='create')
@click.argument('title')
@click.option('--content', '-c', default='', help='Note content')
@click.option('--tags', help='Comma-separated tags')
def notes_create(title, content, tags):
    """Create a note."""
    async def _create():
        nm = NotesManager()
        tag_list = [t.strip() for t in tags.split(',')] if tags else []
        note_id = await nm.create_note(title, content, tag_list)
        console.print(f"[green]Created note: {title} ({note_id[:8]})[/green]")
    
    asyncio.run(_create())


@notes.command(name='list')
def notes_list():
    """List all notes."""
    async def _list():
        nm = NotesManager()
        notes_list = await nm.list_notes()
        if not notes_list:
            console.print("[yellow]No notes[/yellow]")
            return
        table = Table(title="Notes")
        table.add_column("Title", style="cyan")
        table.add_column("Tags", style="magenta")
        table.add_column("Updated", style="dim")
        for n in notes_list:
            tags = ', '.join(n.get('tags', []))
            updated = n.get('updated_at', '')[:10]
            table.add_row(n.get('title', ''), tags, updated)
        console.print(table)
    
    asyncio.run(_list())


@notes.command(name='search')
@click.argument('query')
def notes_search(query):
    """Search notes."""
    async def _search():
        nm = NotesManager()
        results = await nm.search_notes(query)
        if not results:
            console.print("[yellow]No notes found[/yellow]")
            return
        for n in results:
            console.print(Panel(n.get('content', ''), title=f"[bold]{n.get('title')}[/bold]"))
    
    asyncio.run(_search())


@main.group()
def chat_history():
    """Manage chat history."""
    pass


@chat_history.command(name='list')
def chat_list():
    """List conversations."""
    async def _list():
        cm = ChatManager()
        convs = await cm.list_conversations()
        if not convs:
            console.print("[yellow]No conversations[/yellow]")
            return
        table = Table(title="Conversations")
        table.add_column("Title", style="cyan")
        table.add_column("Messages", style="green")
        table.add_column("Updated", style="dim")
        for c in convs:
            updated = c.get('updated_at', '')[:19]
            table.add_row(c.get('title', ''), str(c.get('message_count', 0)), updated)
        console.print(table)
    
    asyncio.run(_list())


@main.command()
def init():
    """Initialize SayLess configuration and directories."""
    settings = get_settings()
    
    console.print(Panel.fit(
        "[bold green]SayLess Initialized[/bold green]",
        border_style="green"
    ))
    
    console.print(f"Data directory: [cyan]{settings.data_dir}[/cyan]")
    console.print(f"Queue directory: [cyan]{settings.queue_dir}[/cyan]")
    console.print(f"Models directory: [cyan]{settings.data_dir / 'models'}[/cyan]")
    
    # Create example .env
    env_path = Path(".env.example")
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write("# SayLess Configuration\n")
            f.write("# Copy to .env and customize\n\n")
            f.write("# STT\n")
            f.write("SAYLESS_STT_ENABLED=true\n")
            f.write("SAYLESS_STT_ENGINE=faster-whisper\n")
            f.write("SAYLESS_STT_MODEL=tiny.en\n")
            f.write("SAYLESS_STT_DEVICE=cpu\n")
            f.write("SAYLESS_STT_AUDIO_DEVICE=hw:1,0\n\n")
            f.write("# LLM (LiteRT-LM / Gemma on CoralBoard)\n")
            f.write("SAYLESS_LLM_ENABLED=true\n")
            f.write("SAYLESS_LLM_ENGINE=litert-lm\n")
            f.write("SAYLESS_LLM_MODEL_NAME=mobile_actions_q8_ekv1024.litertlm\n")
            f.write("SAYLESS_LLM_MODEL_PATH=/home/root/models/mobile_actions_q8_ekv1024.litertlm\n")
            f.write("SAYLESS_LLM_BASE_URL=http://localhost:8000/v1\n")
            f.write("SAYLESS_LLM_SERVER_HOST=127.0.0.1\n")
            f.write("SAYLESS_LLM_SERVER_PORT=8000\n")
            f.write("SAYLESS_LLM_TEMPERATURE=0.7\n")
            f.write("SAYLESS_LLM_TOP_P=0.95\n")
            f.write("SAYLESS_LLM_TOP_K=64\n\n")
            f.write("# Tools\n")
            f.write("SAYLESS_ENABLE_PHONE_TOOLS=true\n")
            f.write("SAYLESS_ENABLE_MAPS_TOOLS=true\n")
            f.write("SAYLESS_GOOGLE_VOICE_ENABLED=false\n")
            f.write("SAYLESS_GOOGLE_MAPS_API_KEY=\n")
        console.print(f"Created [cyan]{env_path}[/cyan]")
    
    console.print("\n[green]Ready![/green] Next: run [cyan]sayless setup[/cyan] to start the LiteRT-LM server.")
    console.print("Then use 'sayless run' or 'sayless chat'.")


if __name__ == '__main__':
    main()
