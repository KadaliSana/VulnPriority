"""``python -m vulnpriority ...`` - the same CLI as the ``vulnpriority`` console script.

The console script is installed into Python's scripts directory, which on Windows is often
not on ``PATH``, so ``vulnpriority serve`` fails with "not recognized as a name of a cmdlet"
even though the package is installed correctly. This module removes that as a class of
problem: ``python -m vulnpriority`` needs nothing but the interpreter that already imported it.
"""

from vulnpriority.cli import main

if __name__ == "__main__":
    main()
