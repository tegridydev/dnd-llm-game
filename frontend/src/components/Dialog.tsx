import { useEffect, useRef, type ReactNode } from "react";

export function Dialog({
  titleId,
  onClose,
  children,
  className = ""
}: {
  titleId: string;
  onClose: () => void;
  children: ReactNode;
  className?: string;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
    const initialFocus = dialog.querySelector<HTMLElement>("[data-dialog-initial-focus]")
      ?? dialog.querySelector<HTMLElement>("[autofocus]")
      ?? dialog.querySelector<HTMLElement>("button, input, textarea, select");
    initialFocus?.focus();
    function onCancel(event: Event) {
      event.preventDefault();
      onCloseRef.current();
    }
    dialog.addEventListener("cancel", onCancel);
    return () => {
      dialog.removeEventListener("cancel", onCancel);
      if (dialog.open && typeof dialog.close === "function") dialog.close();
    };
  }, []);

  return (
    <dialog
      ref={dialogRef}
      className={`dialog-surface ${className}`.trim()}
      aria-labelledby={titleId}
      onMouseDown={(event) => event.currentTarget === event.target && onClose()}
    >
      {children}
    </dialog>
  );
}
