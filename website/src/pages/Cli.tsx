import { createElement, useHead } from "@asymmetric-effort/specifyjs";

export function Cli() {
  useHead({
    title: "CLI Reference — pgmgo",
    description: "Command-line tools for pgmgo probabilistic graphical models.",
    canonical: "https://pgmgo.asymmetric-effort.com/#/cli",
  });

  return (
    <div class="page">
      <h1>CLI Reference</h1>

      <section class="section">
        <h2>Usage</h2>
        <pre><code>pgmgo [command] [options]</code></pre>
      </section>

      <section class="section">
        <h2>Commands</h2>
        <table>
          <thead>
            <tr><th>Command</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>validate &lt;file&gt;</code></td><td>Validate a BIF model file</td></tr>
            <tr><td><code>version</code></td><td>Print version information</td></tr>
            <tr><td><code>help</code></td><td>Show help message</td></tr>
          </tbody>
        </table>
      </section>

      <section class="section">
        <h2>Options</h2>
        <table>
          <thead>
            <tr><th>Flag</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>-h, --help</code></td><td>Show help</td></tr>
            <tr><td><code>-v, --version</code></td><td>Show version</td></tr>
          </tbody>
        </table>
      </section>

      <section class="section">
        <h2>Exit Codes</h2>
        <table>
          <thead>
            <tr><th>Code</th><th>Meaning</th></tr>
          </thead>
          <tbody>
            <tr><td><code>0</code></td><td>Success</td></tr>
            <tr><td><code>1</code></td><td>General error</td></tr>
            <tr><td><code>2</code></td><td>Invalid input</td></tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
