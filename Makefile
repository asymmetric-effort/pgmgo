SHELL := /bin/bash
.PHONY: all build test test-unit test-integration test-e2e lint fmt clean \
        coverage website website-dev generate-fixtures verify-fixtures \
        install-hooks install-skills

SKILLS_VERSION := v0.0.28
COVERAGE_THRESHOLD := 0

all: lint test build

# --- Build ---

build:
	@for cmd in cmd/*/; do \
		name=$$(basename "$$cmd"); \
		echo "Building $$name..."; \
		go build -o "bin/$$name" "./$$cmd"; \
	done

# --- Linting ---

lint:
	go vet ./...
	@unformatted=$$(gofmt -l .); \
	if [ -n "$$unformatted" ]; then \
		echo "Files not formatted:"; echo "$$unformatted"; exit 1; \
	fi

fmt:
	gofmt -w .

# --- Testing ---

test: test-unit

test-unit:
	go test -tags unit -race -count=1 ./...

test-integration:
	go test -tags integration -race -count=1 ./...

test-e2e:
	go test -tags e2e -race -count=1 ./...

test-all:
	go test -tags "unit integration e2e" -race -count=1 ./...

coverage:
	go test -tags unit -race -coverprofile=coverage.out ./...
	@total=$$(go tool cover -func=coverage.out | grep total | awk '{print $$3}' | sed 's/%//'); \
	echo "Total coverage: $${total}%"; \
	threshold=$(COVERAGE_THRESHOLD); \
	if [ $$(echo "$$total < $$threshold" | bc -l) -eq 1 ]; then \
		echo "ERROR: Coverage $${total}% is below threshold $${threshold}%"; \
		exit 1; \
	fi

# --- Test Fixtures ---

generate-fixtures:
	cd tests/python && python3 generate_fixtures.py --all --output ../fixtures

verify-fixtures:
	@echo "Regenerating fixtures to check for drift..."
	@cd tests/python && python3 generate_fixtures.py --all --output ../fixtures.tmp
	@diff -r tests/fixtures tests/fixtures.tmp > /dev/null 2>&1 && \
		echo "Fixtures are up to date." && rm -rf tests/fixtures.tmp || \
		(echo "ERROR: Fixtures are out of date. Run 'make generate-fixtures'."; rm -rf tests/fixtures.tmp; exit 1)

# --- Website ---

website:
	cd website && node node_modules/.bin/vite build

website-dev:
	cd website && node node_modules/.bin/vite

# --- Setup ---

install-hooks:
	ln -sf ../../git-hooks/pre-commit .git/hooks/pre-commit
	ln -sf ../../git-hooks/pre-push .git/hooks/pre-push
	@echo "Git hooks installed."

install-skills:
	curl -fsSL https://skills.asymmetric-effort.com/install.sh | sh -s $(SKILLS_VERSION)

# --- Clean ---

clean:
	rm -rf bin/ website/dist/ coverage.out
