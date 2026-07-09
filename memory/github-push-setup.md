---
name: github-push-setup
description: How to push this repo to GitHub from the Argon cluster (dedicated SSH key + config)
metadata: 
  node_type: memory
  type: reference
  originSessionId: a191fb17-15d6-48bc-be53-f9709388017a
---

Repo `origin` = `git@github.com:iampartho/UTE_LUNG_SEGMENTATION_PHASE_2.git` (user: iampartho).

Pushing works via a **dedicated GitHub SSH key** set up 2026-07-09:
- Private key: `~/.ssh/github_iampartho` (no passphrase, so pushes are non-interactive).
- `~/.ssh/config` has a `Host github.com` block pinning `IdentityFile ~/.ssh/github_iampartho` with `IdentitiesOnly yes`.
- The cluster's default key `~/.ssh/cluster` ("Warewulf Cluster key") is NOT registered on GitHub — do not rely on it for github.
- Just run `git push origin main`; no token needed.

Repo `.gitignore` ignores ALL non-`.py` files and all subdirectories, but a trailing `!*.py` rule re-includes anything ending in `.py` — including macOS `._*.py` AppleDouble junk. So `git add -A` will stage `._*.py` files; unstage them with `git reset HEAD -- '._*.py'`. PNGs/plots must be force-added (`git add -f`). See [[ute-energy-guidance-project]].
