"""Template for `local_secrets.py`. This file IS committed.

Copy it to `local_secrets.py` in the same directory and put a real key in the copy:

    copy wilcom_pipeline\\local_secrets.example.py wilcom_pipeline\\local_secrets.py

`config.api_key()` then finds the key with no environment variable and no flag.
The real file is gitignored, so the key never reaches the repository. The two files
are separate precisely so that this one can be public and that one cannot.

Prefer the environment for a shared or CI machine, where a file on disk is a worse
place for a secret:

    $env:GEMINI_API_KEY = 'AQ...'        # PowerShell
    export GEMINI_API_KEY=AQ...          # bash
"""

# Replace with a real key from https://aistudio.google.com/apikey
GEMINI_API_KEY = ""
