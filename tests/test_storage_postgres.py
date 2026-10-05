"""The schema: the migrations and dclab_rnd/storage/models.py describe the same database (package 9.2)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pgtest  # noqa: E402


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.url = pgtest.require()

    def test_the_migrations_produce_exactly_the_modelled_schema(self):
        from alembic.autogenerate import compare_metadata
        from alembic.migration import MigrationContext

        from dclab_rnd.storage import db
        from dclab_rnd.storage.models import metadata

        with db.engine(self.url).connect() as connection:
            differences = compare_metadata(MigrationContext.configure(connection, opts={"compare_type": True}), metadata)
        self.assertEqual(differences, [], "models.py and the migrations disagree: write a migration (alembic revision)")

    def test_the_database_is_at_the_latest_revision_and_upgrade_is_repeatable(self):
        from alembic.script import ScriptDirectory

        from dclab_rnd.storage import db

        head = ScriptDirectory.from_config(db._config(self.url)).get_current_head()
        self.assertEqual(db.current(self.url), head)
        db.upgrade(self.url)  # running it again changes nothing
        self.assertEqual(db.current(self.url), head)

    def test_a_downgrade_and_upgrade_round_trip_on_a_scratch_database(self):
        import sqlalchemy as sa

        from dclab_rnd.storage import db

        admin = sa.create_engine(self.url.rsplit("/", 1)[0] + "/postgres", isolation_level="AUTOCOMMIT")
        scratch = "dclab_scratch_migrations"
        url = self.url.rsplit("/", 1)[0] + "/" + scratch
        with admin.connect() as c:
            c.execute(sa.text(f"drop database if exists {scratch}"))
            c.execute(sa.text(f"create database {scratch}"))
        try:
            db.upgrade(url)
            db.upgrade(url, "head")
            from alembic import command
            command.downgrade(db._config(url), "base")
            self.assertIsNone(db.current(url))
            db.upgrade(url)
            self.assertIsNotNone(db.current(url))
        finally:
            db.engine(url).dispose()
            with admin.connect() as c:
                c.execute(sa.text(f"drop database if exists {scratch} with (force)"))
            admin.dispose()

    def test_every_list_is_served_by_an_index(self):
        import sqlalchemy as sa

        from dclab_rnd.storage import db

        with db.engine(self.url).connect() as c:
            names = {r[0] for r in c.execute(sa.text("select indexname from pg_indexes where schemaname = 'public'"))}
        for wanted in ("ix_project_workspace_updated", "ix_draft_workspace_updated", "ix_intern_session_workspace_updated", "ix_activity_project_id", "ix_transition_project_id"):
            self.assertIn(wanted, names)


if __name__ == "__main__":
    unittest.main()
