"""python -m dclab_core.prompts [--record]: print the prompts' fingerprints, or record them in HASHES.json."""
import json
import sys

from dclab_core.prompts import HASHES, fingerprints

if "--record" in sys.argv:
    HASHES.write_text(json.dumps(fingerprints(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(fingerprints(), indent=2, sort_keys=True))
