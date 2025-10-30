#!/usr/bin/env python3
"""
Health monitoring CLI tool.

This tool provides command-line access to the health monitoring system.
"""

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import click
from src.common.health_monitor import get_health_monitor
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@click.group()
def cli():
    """Health monitoring CLI tool."""
    pass


@cli.command()
@click.option('--system', help='Check specific system only')
@click.option('--json-output', is_flag=True, help='Output in JSON format')
async def check(system, json_output):
    """Check system health."""
    try:
        health_monitor = get_health_monitor()

        if system:
            result = await health_monitor.check_system_health(system)
            if json_output:
                output = {
                    "system": result.system,
                    "component": result.component,
                    "status": result.status.value,
                    "response_time_ms": result.response_time_ms,
                    "error_message": result.error_message,
                    "metadata": result.metadata
                }
                click.echo(json.dumps(output, indent=2))
            else:
                click.echo(f"System: {result.system}")
                if result.component:
                    click.echo(f"Component: {result.component}")
                click.echo(f"Status: {result.status.value}")
                if result.response_time_ms:
                    click.echo(f"Response Time: {result.response_time_ms}ms")
                if result.error_message:
                    click.echo(f"Error: {result.error_message}")
        else:
            results = await health_monitor.check_all_systems_health()

            if json_output:
                output = []
                for result in results:
                    output.append({
                        "system": result.system,
                        "component": result.component,
                        "status": result.status.value,
                        "response_time_ms": result.response_time_ms,
                        "error_message": result.error_message,
                        "metadata": result.metadata
                    })
                click.echo(json.dumps(output, indent=2))
            else:
                click.echo("System Health Status:")
                click.echo("-" * 50)
                for result in results:
                    system_name = result.system
                    if result.component:
                        system_name += f".{result.component}"

                    status_color = "green" if result.status.value == "HEALTHY" else \
                                  "yellow" if result.status.value == "DEGRADED" else "red"

                    click.echo(f"{system_name:<20} {click.style(result.status.value, fg=status_color)}")
                    if result.error_message:
                        click.echo(f"  Error: {result.error_message}")
                    if result.response_time_ms:
                        click.echo(f"  Response Time: {result.response_time_ms}ms")

    except Exception as e:
        click.echo(f"Error checking health: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--json-output', is_flag=True, help='Output in JSON format')
async def summary(json_output):
    """Get health summary from database."""
    try:
        health_monitor = get_health_monitor()
        summary = await health_monitor.get_system_health_summary()

        if json_output:
            click.echo(json.dumps(summary, indent=2))
        else:
            click.echo(f"Overall Status: {click.style(summary['overall_status'], fg='green' if summary['overall_status'] == 'HEALTHY' else 'red')}")
            click.echo(f"Timestamp: {summary['timestamp']}")
            click.echo()

            stats = summary['statistics']
            click.echo("Statistics:")
            click.echo(f"  Total Systems: {stats['total_systems']}")
            click.echo(f"  Healthy: {click.style(str(stats['healthy_systems']), fg='green')}")
            click.echo(f"  Degraded: {click.style(str(stats['degraded_systems']), fg='yellow')}")
            click.echo(f"  Down: {click.style(str(stats['down_systems']), fg='red')}")
            click.echo(f"  Unknown: {stats['unknown_systems']}")
            click.echo()

            click.echo("Systems:")
            for system_name, system_data in summary['systems'].items():
                status_color = "green" if system_data['status'] == "HEALTHY" else \
                              "yellow" if system_data['status'] == "DEGRADED" else "red"

                click.echo(f"  {system_name:<25} {click.style(system_data['status'], fg=status_color)}")
                if system_data['error_message']:
                    click.echo(f"    Error: {system_data['error_message']}")
                if system_data['avg_response_time_ms']:
                    click.echo(f"    Avg Response: {system_data['avg_response_time_ms']}ms")

    except Exception as e:
        click.echo(f"Error getting summary: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--interval', default=60, help='Check interval in seconds')
async def monitor(interval):
    """Start continuous health monitoring."""
    try:
        health_monitor = get_health_monitor()
        click.echo(f"Starting health monitoring with {interval}s interval...")
        click.echo("Press Ctrl+C to stop")

        await health_monitor.start_monitoring(interval)

    except KeyboardInterrupt:
        click.echo("\nStopping health monitoring...")
        health_monitor.stop_monitoring()
    except Exception as e:
        click.echo(f"Error during monitoring: {e}", err=True)
        sys.exit(1)


@cli.command()
async def migrate():
    """Run the database migration from channel health to system health."""
    try:
        from src.data.db.migrations.convert_channel_health_to_system_health import migrate_channel_health_to_system_health

        click.echo("Running migration from msg_channel_health to msg_system_health...")
        migrate_channel_health_to_system_health()
        click.echo("Migration completed successfully!")

    except Exception as e:
        click.echo(f"Migration failed: {e}", err=True)
        sys.exit(1)


@cli.command()
async def rollback():
    """Rollback the database migration."""
    try:
        from src.data.db.migrations.convert_channel_health_to_system_health import rollback_migration

        if click.confirm("Are you sure you want to rollback the migration?"):
            click.echo("Rolling back migration...")
            rollback_migration()
            click.echo("Rollback completed successfully!")
        else:
            click.echo("Rollback cancelled.")

    except Exception as e:
        click.echo(f"Rollback failed: {e}", err=True)
        sys.exit(1)


def async_command(f):
    """Decorator to run async commands."""
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper


# Apply async decorator to async commands
check = async_command(check)
summary = async_command(summary)
monitor = async_command(monitor)
migrate = async_command(migrate)
rollback = async_command(rollback)


if __name__ == '__main__':
    cli()