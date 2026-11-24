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
    const returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
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
      if (returnFocus?.isConnected && document.activeElement !== returnFocus) returnFocus.focus();
    };
  }, []);

  return (
    <dialog
      ref={dialogRef}
      className={`dialog-surface ${className}`.trim()}
      aria-labelledby={titleId}
      onClick={(event) => {
        if (event.currentTarget !== event.target) return;
        const bounds = event.currentTarget.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right ||
          event.clientY < bounds.top || event.clientY > bounds.bottom) onClose();
      }}
    >
      {children}
    </dialog>
  );
}
