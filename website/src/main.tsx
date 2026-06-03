import { createElement, Router, Route } from "@asymmetric-effort/specifyjs";
import { createRoot } from "@asymmetric-effort/specifyjs/dom";
import { Nav } from "./components/Nav";
import { Footer } from "./components/Footer";
import { Home } from "./pages/Home";
import { Docs } from "./pages/Docs";
import { Cli } from "./pages/Cli";
import { Api } from "./pages/Api";
import { Tutorials } from "./pages/Tutorials";
import "./styles.css";

function App() {
  return (
    <Router>
      <div id="app">
        <Nav />
        <main>
          <Route path="/" component={Home} />
          <Route path="/docs" component={Docs} />
          <Route path="/cli" component={Cli} />
          <Route path="/api" component={Api} />
          <Route path="/tutorials" component={Tutorials} />
        </main>
        <Footer />
      </div>
    </Router>
  );
}

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(<App />);
}
