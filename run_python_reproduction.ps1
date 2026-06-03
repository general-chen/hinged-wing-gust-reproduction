param(
    [string]$Python = "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)

$ErrorActionPreference = "Stop"

& $Python (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "reproduce_from_description.py")
