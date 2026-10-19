#!/bin/sh
# Drop to appuser, keeping only CAP_SETUID/CAP_SETGID (ambient) so the orchestrator can
# start opencode as the `agent` user. The child gets no capabilities (runner._priv_drop).
set -e
exec setpriv --reuid=appuser --regid=appuser --init-groups \
    --inh-caps=-all,+setuid,+setgid --ambient-caps=+setuid,+setgid -- "$@"
