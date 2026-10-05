"""Alembic environment: the database URL comes from the caller (``dclab_rnd.storage.db.upgrade``) or DCLAB_DATABASE_URL."""
from alembic import context
from sqlalchemy import create_engine

from dclab_rnd.storage import db
from dclab_rnd.storage.models import metadata

config = context.config


def run() -> None:
    url = config.attributes.get("url") or db.database_url()
    connection = config.attributes.get("connection")
    engine = None
    if connection is None:
        engine = create_engine(url)
        connection = engine.connect()
    try:
        context.configure(connection=connection, target_metadata=metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
        if engine is not None:
            connection.commit()
    finally:
        if engine is not None:
            connection.close()
            engine.dispose()


run()
