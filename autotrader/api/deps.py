"""Things routes need, as dependencies tests can override."""

from functools import lru_cache

from autotrader.data.archive import BarArchive


def get_sessions():
    from autotrader.db.session import session_factory

    return session_factory()


@lru_cache
def _archive() -> BarArchive:
    return BarArchive()


def get_archive():
    return _archive()


def get_arena():
    """The paper arena (broker keys from .env), built only for kill/unkill."""
    from autotrader.cli import _arena

    return _arena()


def arena_factory():
    """Returns a callable, so the arena (and its broker connection) is built only when used."""
    return get_arena


def get_agents_cfg():
    from autotrader.orchestrator.config import load_agents_config

    return load_agents_config()


def get_policy():
    from autotrader.evaluator.policy import load_policy

    return load_policy()
