"""Single source of truth for the app version. Read by build.ps1 (regex on
VERSION) and shown in the GUI title / about dialog. Bump per PACKAGING.md
section 2 (SemVer) when cutting a release -- nothing else reads git tags.
"""

VERSION = "1.1.2"
