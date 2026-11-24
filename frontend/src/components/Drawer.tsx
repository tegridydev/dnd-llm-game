import { X } from "lucide-react";
import { useId, type ReactNode } from "react";
import { Dialog } from "./Dialog";

export function Drawer({ title, eyebrow, onClose, children, side = "right" }: {
  title: string;
  eyebrow?: string;
  onClose: () => void;
  children: ReactNode;
  side?: "left" | "right";
}) {
  const titleId = useId();
  return (
    <Dialog titleId={titleId} onClose={onClose} className={`app-drawer ${side}`}>
      <header className="drawer-header">
        <div>
          {eyebrow && <span className="eyebrow">{eyebrow}</span>}
          <h2 id={titleId}>{title}</h2>
        </div>
        <button type="button" className="icon-button" onClick={onClose} aria-label={`Close ${title}`}>
          <X size={18} aria-hidden="true" />
        </button>
      </header>
      <div className="drawer-content">{children}</div>
    </Dialog>
  );
}
