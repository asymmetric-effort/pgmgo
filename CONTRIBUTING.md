# Contributing to pgmgo

Thank you for your interest in contributing to pgmgo.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Install git hooks: `make install-hooks`
4. Create a feature branch from `main`
5. Make your changes
6. Run tests: `make test`
7. Commit your changes with a clear commit message (Conventional Commits format)
8. Push to your fork and open a pull request

## Git Hooks

Install pre-commit and pre-push hooks:

```bash
make install-hooks
```

- **pre-commit**: runs `gofmt` check, `go vet`, and secret scanning
- **pre-push**: runs unit tests with race detection

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

- `feat:` — new feature
- `fix:` — bug fix
- `test:` — adding or updating tests
- `docs:` — documentation changes
- `refactor:` — code restructuring without behavior change
- `perf:` — performance improvement
- `chore:` — maintenance tasks

## Testing

pgmgo uses Python + pgmpy as a reference oracle for cross-validation testing.

```bash
make test-unit          # Run unit tests
make test-integration   # Run integration tests
make test-all           # Run all test tiers
make coverage           # Run tests with coverage report
make generate-fixtures  # Regenerate Python test fixtures
```

### Test Tiers

Tests use Go build tags:

- `//go:build unit` — fast, no external dependencies
- `//go:build integration` — may require Python/pgmpy
- `//go:build e2e` — end-to-end tests

### Test Fixtures

Test fixtures are generated from pgmpy and stored in `tests/fixtures/`.
To regenerate: `make generate-fixtures` (requires Python 3.12+ with pgmpy).

## Guidelines

- Follow standard Go conventions and `gofmt` formatting
- Write tests for new functionality
- Keep pull requests focused on a single change
- Update documentation as needed

## Dependencies

pgmgo maintains a near-zero third-party dependency policy to minimize supply-chain
risk. Do not add external dependencies without prior discussion and explicit
approval from maintainers.

## Code Review

All submissions require review before merging. Maintainers may request changes
or improvements before a pull request is accepted.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License as described in [LICENSE.txt](LICENSE.txt).
