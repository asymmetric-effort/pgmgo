import { createElement, Link, useRouter } from "@asymmetric-effort/specifyjs";

export function Nav() {
  const { pathname } = useRouter();

  return (
    <nav class="nav">
      <div class="nav-inner">
        <Link to="/" class="nav-brand">pgmgo</Link>
        <div class="nav-links">
          <Link to="/" class={pathname === "/" ? "active" : ""}>Home</Link>
          <Link to="/docs" class={pathname === "/docs" ? "active" : ""}>Docs</Link>
          <Link to="/tutorials" class={pathname === "/tutorials" ? "active" : ""}>Tutorials</Link>
          <Link to="/cli" class={pathname === "/cli" ? "active" : ""}>CLI</Link>
          <Link to="/api" class={pathname === "/api" ? "active" : ""}>API</Link>
        </div>
      </div>
    </nav>
  );
}
