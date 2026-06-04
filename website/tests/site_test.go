//go:build e2e

package tests

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestWebsiteLinks(t *testing.T) {
	// Serve the built dist directory
	distDir := filepath.Join("..", "dist")
	if _, err := os.Stat(distDir); os.IsNotExist(err) {
		t.Skip("dist/ not found — run 'npm run build' first")
	}

	fs := http.FileServer(http.Dir(distDir))
	server := httptest.NewServer(fs)
	defer server.Close()

	// Fetch index.html
	resp, err := http.Get(server.URL + "/")
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("/ returned %d", resp.StatusCode)
	}

	body, _ := io.ReadAll(resp.Body)
	html := string(body)

	// Verify root element exists
	if !strings.Contains(html, `<div id="root">`) {
		t.Error("missing root element")
	}

	// Verify JS bundle loaded
	if !strings.Contains(html, "assets/index-") {
		t.Error("missing JS bundle reference")
	}

	// Verify static assets are accessible
	assets := []string{
		"/docs/img/logo.png",
		"/CNAME",
		"/robots.txt",
		"/sitemap.xml",
		"/404.html",
	}
	for _, asset := range assets {
		r, err := http.Get(server.URL + asset)
		if err != nil {
			t.Errorf("%s: %v", asset, err)
			continue
		}
		if r.StatusCode != 200 {
			t.Errorf("%s returned %d", asset, r.StatusCode)
		}
	}

	// Extract JS bundle URL and fetch it
	re := regexp.MustCompile(`src="/assets/(index-[^"]+\.js)"`)
	matches := re.FindStringSubmatch(html)
	if len(matches) < 2 {
		t.Fatal("can't find JS bundle URL")
	}

	jsResp, _ := http.Get(server.URL + "/assets/" + matches[1])
	if jsResp.StatusCode != 200 {
		t.Fatal("JS bundle not accessible")
	}
	jsBody, _ := io.ReadAll(jsResp.Body)
	js := string(jsBody)

	// Verify all routes are defined in the JS bundle
	routes := []string{"/", "/docs", "/cli", "/api", "/tutorials"}
	for _, route := range routes {
		// Routes appear as string literals in the bundle (double, single, or backtick quotes)
		if !strings.Contains(js, `"`+route+`"`) &&
			!strings.Contains(js, `'`+route+`'`) &&
			!strings.Contains(js, "`"+route+"`") {
			t.Errorf("route %s not found in JS bundle", route)
		}
	}

	// Verify page content strings are in the JS bundle
	pageContent := map[string][]string{
		"Home":      {"Probabilistic Graphical Models", "Installation", "Quick Start", "Features"},
		"Docs":      {"Documentation", "Getting Started", "Library Packages", "Core Packages"},
		"CLI":       {"CLI Reference", "validate", "query", "learn", "sample", "convert"},
		"API":       {"API Reference", "BayesianNetwork", "VariableElimination", "Import Paths"},
		"Tutorials": {"Tutorials", "Building Your First", "Learning Structure", "Causal Inference"},
	}
	for page, keywords := range pageContent {
		for _, kw := range keywords {
			if !strings.Contains(js, kw) {
				t.Errorf("page %s: missing content keyword %q in bundle", page, kw)
			}
		}
	}

	// Verify sitemap lists all routes
	sitemapResp, _ := http.Get(server.URL + "/sitemap.xml")
	sitemapBody, _ := io.ReadAll(sitemapResp.Body)
	sitemap := string(sitemapBody)
	for _, route := range routes {
		hashRoute := "/#" + route
		if route == "/" {
			hashRoute = "/"
		}
		if !strings.Contains(sitemap, hashRoute) && !strings.Contains(sitemap, "pgmgo.asymmetric-effort.com"+route) {
			t.Errorf("sitemap missing route %s", route)
		}
	}

	// Verify 404.html has redirect script
	fourOhFourResp, _ := http.Get(server.URL + "/404.html")
	fourOhFourBody, _ := io.ReadAll(fourOhFourResp.Body)
	if !strings.Contains(string(fourOhFourBody), "window.location.replace") {
		t.Error("404.html missing redirect script")
	}
}

func TestWebsiteCSS(t *testing.T) {
	distDir := filepath.Join("..", "dist")
	if _, err := os.Stat(distDir); os.IsNotExist(err) {
		t.Skip("dist/ not found")
	}

	fs := http.FileServer(http.Dir(distDir))
	server := httptest.NewServer(fs)
	defer server.Close()

	resp, _ := http.Get(server.URL + "/")
	body, _ := io.ReadAll(resp.Body)
	html := string(body)

	re := regexp.MustCompile(`href="/assets/(index-[^"]+\.css)"`)
	matches := re.FindStringSubmatch(html)
	if len(matches) < 2 {
		t.Fatal("no CSS bundle found")
	}

	cssResp, _ := http.Get(server.URL + "/assets/" + matches[1])
	if cssResp.StatusCode != 200 {
		t.Fatal("CSS not accessible")
	}
	cssBody, _ := io.ReadAll(cssResp.Body)
	css := string(cssBody)

	// Verify key CSS rules exist
	cssChecks := []string{"--primary", "--bg", "--text", ".nav", ".footer", ".page", ".section"}
	for _, check := range cssChecks {
		if !strings.Contains(css, check) {
			t.Errorf("CSS missing %q", check)
		}
	}
}
