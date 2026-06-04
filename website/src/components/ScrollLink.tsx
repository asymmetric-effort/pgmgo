import { createElement } from "@asymmetric-effort/specifyjs";

// ScrollLink provides in-page anchor navigation that works with
// hash-based SPA routing. Instead of using <a href="#id"> which
// would break the route, it uses scrollIntoView on click.
export function ScrollLink({ to, children, class: className }: { to: string; children: any; class?: string }) {
  const handleClick = (e: Event) => {
    e.preventDefault();
    const target = document.getElementById(to);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <a href="javascript:void(0)" onClick={handleClick} class={className || ""}>
      {children}
    </a>
  );
}
