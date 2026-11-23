# Contributing to iletki

Thank you for helping. This guide covers how changes are made; [AGENTS.md](AGENTS.md) holds the
rules the code follows.

## Setup

```sh
rustup target add wasm32-unknown-unknown
cargo install --locked cargo-deny
cargo xtask ci
```

`cargo xtask ci` runs formatting, clippy, tests, documentation, the dependency policy and a web
build. It must pass before a pull request is reviewed.

## Branches

Work on a short-lived branch named after the kind of change:

```text
feat/<topic>   fix/<topic>   perf/<topic>   docs/<topic>   refactor/<topic>   test/<topic>
build/<topic>  ci/<topic>    chore/<topic>
```

`main` always passes `cargo xtask ci`. Branches are merged with a merge commit (`--no-ff`).

## Commit messages

iletki uses [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<scope>): <summary in the imperative, lower case, no full stop>

<optional body: why the change was needed, wrapped at 72 columns>
```

- Keep the subject around 50 characters (72 at most).
- Add a body only when the reason is not obvious from the subject.
- Types: `feat`, `fix`, `perf`, `docs`, `refactor`, `test`, `build`, `ci`, `chore`.
- Scopes: a crate without the `iletki-` prefix (`core`, `wgpu`, `2d`, `3d`, `egui`, `xtask`).
- Breaking changes: `!` after the type and a `BREAKING CHANGE:` footer.
- One logical change per commit; tests land with the change they cover.

## Releasing

Update the workspace version, internal dependency requirements, standalone example
requirements, lockfiles and [changelog](CHANGELOG.md) together. Run `cargo xtask ci`
and `python3 .github/scripts/package-smoke.py` before publishing.

Publish the crates with `cargo publish --locked -p <crate>` from the release commit.
Publish `iletki-core` first, then `iletki-wgpu`. Next publish `iletki-2d`, `iletki-3d`,
`iletki-egui`, and `iletki-slint`. Publish the `iletki` facade last. Wait for each
dependency to be visible in the registry before publishing a crate that uses it.

Push the matching `v<version>` tag from the release commit. The CI workflow publishes
the GitHub Release with the corresponding changelog entry after its checks pass.

## License

Unless you state otherwise, any contribution you submit is dual licensed under the MIT license and
the Apache License 2.0, as defined in the Apache-2.0 license, without any additional terms or
conditions.
