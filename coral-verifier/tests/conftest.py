import os, sys
here = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(here, "..", "verifier"), "/app"):
    if os.path.isdir(p):
        sys.path.insert(0, p)
