"""``python -m vulnprio ...`` - the same CLI as the ``vulnprio`` console script.

The console script is installed into Python's scripts directory, which on Windows is often
not on ``PATH``, so ``vulnprio serve`` fails with "not recognized as a name of a cmdlet"
even though the package is installed correctly. This module removes that as a class of
problem: ``python -m vulnprio`` needs nothing but the interpreter that already imported it.
"""

from vulnprio.cli import main

if __name__ == "__main__":
    main()
