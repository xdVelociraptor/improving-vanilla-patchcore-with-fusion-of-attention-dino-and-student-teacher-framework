# Optional original Windows environment archive

The [Windows environment release](https://github.com/xdVelociraptor/improving-vanilla-patchcore-with-fusion-of-attention-dino-and-student-teacher-framework/releases/tag/windows-environment-2026-11) preserves all **38,386 original `.venv` files** in four independent ZIP archives. It is about **3.2 GB compressed**, about **5.3 GB extracted**, and includes the installed packages' licenses and notices.

Every archived file matched its original SHA-256. The [archive manifest](environment-archive-manifest.json) records ZIP hashes, byte sizes and file counts.

For archival recovery:

1. Download all four numbered ZIP files and `environment-manifest.json` from the release.
2. Check each ZIP's SHA-256 against the manifest or `SHA256SUMS.txt`. In PowerShell, use `Get-FileHash -Algorithm SHA256 <archive.zip>`.
3. Extract all four ZIPs into the **same separate archival directory**. Each archive adds files beneath `.venv`. Do not concatenate the ZIPs and do not overwrite a working environment.

This is a Windows/Python 3.12.10 snapshot, **not a portable Python installation**. It retains original absolute paths and editable-install references. The base Python installation and NVIDIA driver are separate prerequisites. Moving the archive to another machine does not guarantee that its Python launcher or installed command wrappers will work.

For running the project on a new PC, use `setup_online.ps1` from the repository root. It rebuilds the environment from the pinned requirements. The optional environment archive is intentionally excluded from automatic setup downloads.

Imagenette, WideResNet weights and the EfficientAD teacher remain publisher downloads. Redistribution permission for those inputs has not been established for this repository, so they are not mirrored here. Their verified downloads are already included in online setup.
