# SPDX-License-Identifier: Apache-2.0
"""Root conftest — prevent pytest from importing the ComfyUI entry-point
__init__.py at the worktree root, which pulls in torch/ComfyUI deps.

Used by tests that run outside the e2e container (no torch).
"""
collect_ignore_glob = ["__init__.py", "conftest.py"]
