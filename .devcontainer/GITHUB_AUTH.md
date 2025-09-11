GitHub Authentication in the Dev Container

This container includes Git and the GitHub CLI (`gh`) to make GitHub access reliable. Choose one of the methods below and follow the steps from inside the container terminal.

1) GitHub CLI (HTTPS; recommended and simplest)

- Login: `gh auth login --hostname github.com --git-protocol https --web`
  - If the container has no browser, use: `gh auth login --hostname github.com --git-protocol https --device`
- Configure Git to use your `gh` token: `gh auth setup-git`
- Verify: `gh auth status` and `git ls-remote https://github.com/<owner>/<repo>.git`.

Notes
- The token is stored only in the container under `~/.config/gh/hosts.yml`.
- This avoids storing Personal Access Tokens in plain files or env vars.

2) SSH with agent forwarding (use if you already use SSH keys)

- Ensure your host has an SSH key added to GitHub and the agent loaded (`ssh-add -l`).
- Rebuild/reopen the dev container. The configuration includes the `ssh-agent` feature.
- Verify inside the container: `ssh -T git@github.com` (expect a success greeting).
- Use SSH remotes (recommended for SSH): `git remote set-url origin git@github.com:<owner>/<repo>.git`

Notes
- Your private keys do not need to be copied into the container; the agent forwards signatures.
- On Windows, ensure the OpenSSH agent is running and the key is added on the host.

3) Environment token passthrough (for CI or advanced users)

- If you set `GH_TOKEN` or `GITHUB_TOKEN` on the host, the dev container forwards them.
- Verify inside the container: `echo $GH_TOKEN` or `echo $GITHUB_TOKEN` (should be non-empty if set on host).
- You can point Git to use the token by running `gh auth login --with-token` and pasting the env var value, or by using `git` with HTTPS URLs (the `gh` credential helper will handle it after `gh auth setup-git`).

Quick Checks
- `gh auth status` shows current auth state.
- `git config --global --get credential.helper` should include the `gh` helper after `gh auth setup-git`.
- `git remote -v` confirms whether you use HTTPS or SSH URLs.

Security Tips
- Prefer `gh auth login` over manually creating/storing PATs.
- Avoid mounting your `~/.ssh` into the container; agent forwarding is safer.
- Rotate tokens/keys periodically and revoke unused credentials from your GitHub account settings.
