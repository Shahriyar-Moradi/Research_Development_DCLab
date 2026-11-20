"""The work passed between team agents, kept and delivered (W4 of the collaborative workspace, 2026-11-20).

    store.py     tasks, actions, reports and the delivery outbox: files in the workspace folder, or PostgreSQL rows
                 (migration 0026). Every change goes through dclab_core.team_work's rules.
    delivery.py  the ``team_delivery`` job: reads the saved outbox row, rechecks the member who started it, hands it to
                 a pluggable target (W5's agents), marks it delivered, retries a temporary failure a bounded number of
                 times, and puts every undelivered row back in the queue when a worker starts.

The records are ``dclab_core.team_work``'s (W1); the screens that read them are W8's.
"""

from .store import Conflict, FileTeamWork, PgTeamWork, Written, history_lines, open_team_work, team_label

__all__ = ["Conflict", "FileTeamWork", "PgTeamWork", "Written", "history_lines", "open_team_work", "team_label"]
