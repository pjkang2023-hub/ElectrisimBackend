# -*- coding: utf-8 -*-
"""
Gunicorn configuration for the Electrisim simulation API.

`gunicorn app:app` with no configuration runs ONE synchronous worker with a
30-second timeout. For this service that means:

  - a single contingency sweep or transient-stability run blocks every other
    user for its whole duration, and
  - anything slower than 30 s is killed mid-solve and returned as a worker
    timeout, which the frontend surfaces as a generic network error.

Parallelism here has to come from processes, not threads: opendss_electrisim
drives the opendssdirect singleton, which is process-global, and serialises
access with `_opendss_engine_lock`. Extra threads in one worker would queue on
that lock and gain nothing; separate worker processes each get their own
OpenDSS engine. So: sync workers, more than one, long timeout.

Usage:  gunicorn -c gunicorn.conf.py app:app
"""

import multiprocessing
import os


def _int_env(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"

# Sync workers: each handles one request at a time, which matches CPU-bound
# solves and keeps the OpenDSS engine single-user per process.
worker_class = 'sync'

# Each worker holds a full pandapower/numpy/OpenDSS stack, so these are
# memory-hungry - this is not a "2x cores + 1" web app. Default to a small
# number and let the platform override via WEB_CONCURRENCY.
workers = _int_env('WEB_CONCURRENCY', min(4, max(2, multiprocessing.cpu_count() // 2)))

# Long solves must not be killed. Raise further if your largest study needs it;
# this is the ceiling on how long one worker can be monopolised.
timeout = _int_env('GUNICORN_TIMEOUT', 600)
graceful_timeout = _int_env('GUNICORN_GRACEFUL_TIMEOUT', 30)

# Hold idle upstream connections open a little longer than a typical proxy.
keepalive = _int_env('GUNICORN_KEEPALIVE', 10)

# Recycle workers periodically. Long-lived processes doing heavy numpy and
# OpenDSS work accumulate memory; jitter avoids restarting them in lockstep.
max_requests = _int_env('GUNICORN_MAX_REQUESTS', 200)
max_requests_jitter = _int_env('GUNICORN_MAX_REQUESTS_JITTER', 50)

# Cap queued connections so overload fails fast instead of building a backlog
# that is already past its client timeout by the time a worker picks it up.
backlog = _int_env('GUNICORN_BACKLOG', 64)

# NOT preloading: opendssdirect initialises process-global engine state at
# import. Forking after that would hand every worker a copy of the same
# initialised engine. Each worker imports its own.
preload_app = False

accesslog = '-'
errorlog = '-'
loglevel = os.getenv('GUNICORN_LOG_LEVEL', 'info')
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(M)sms'


def on_starting(server):
    server.log.info(
        'Electrisim: %s sync workers, %ss timeout, recycle every ~%s requests',
        workers, timeout, max_requests
    )
