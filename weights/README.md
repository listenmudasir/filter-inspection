# weights/

`supervised_global.pt` (262 MiB) is **not** tracked in git: it exceeds GitHub's
100 MB per-file hard limit, and copying it directly avoids needing Git LFS.

Copy it onto each machine after cloning:

    cp /path/to/supervised_global.pt  weights/

`../selftest.py` fails loudly if it is missing or truncated.
