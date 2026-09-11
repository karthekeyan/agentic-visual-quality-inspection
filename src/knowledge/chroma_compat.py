"""Import this before `chromadb` anywhere in the project.

The system's sqlite3 (3.31.1) is older than what chromadb requires
(>= 3.35.0). pysqlite3-binary ships a modern sqlite3 build; swapping it
into sys.modules['sqlite3'] before chromadb is imported is the fix
chromadb's own troubleshooting docs recommend, since chromadb imports the
stdlib sqlite3 module by name internally and can't be pointed at a
different module via its own API.
"""

import sys

try:
    import pysqlite3  # noqa: F401

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # system sqlite3 may already be new enough
