import { BookOpen, FileText, LoaderCircle, RefreshCw, Trash2, Upload, X } from "lucide-react";
import { useRef, useState } from "react";
import { Dialog } from "../../components/Dialog";
import type { LoreDocument, WorkerStatus } from "../../types";

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

export function LorePanel({
  lore,
  worker,
  onRefresh,
  onUpload,
  onDelete
}: {
  lore: LoreDocument[];
  worker: WorkerStatus | null;
  onRefresh: (force?: boolean) => Promise<void>;
  onUpload: (file: File) => Promise<void>;
  onDelete: (id: number) => Promise<void>;
}) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);

  async function upload(file: File | undefined) {
    if (!file || uploading) return;
    setUploading(true);
    try {
      await onUpload(file);
    } catch {
      // The application-level error banner owns mutation failures.
    } finally {
      if (fileInput.current) fileInput.current.value = "";
      setUploading(false);
    }
  }

  async function refresh(force = false) {
    setRefreshing(true);
    try {
      await onRefresh(force);
    } catch {
      // The application-level error banner owns mutation failures.
    } finally {
      setRefreshing(false);
    }
  }

  return (
    <section className="panel compact lore-panel">
      <div className="panel-title">
        <BookOpen size={16} aria-hidden="true" />
        <span>Lore Library</span>
        <button
          type="button"
          className="icon-button"
          onClick={() => refresh(false)}
          disabled={refreshing}
          aria-label="Queue unfinished lore documents for indexing"
        >
          <RefreshCw className={refreshing ? "spin" : undefined} size={15} aria-hidden="true" />
        </button>
      </div>

      <label className={uploading ? "upload disabled" : "upload"}
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => { event.preventDefault(); void upload(event.dataTransfer.files?.[0]); }}>
        {uploading ? <LoaderCircle className="spin" size={18} /> : <Upload size={18} />}
        <span>{uploading ? "Storing PDF..." : "Choose or drop a lore PDF"}</span>
        <input
          ref={fileInput}
          type="file"
          accept="application/pdf,.pdf"
          disabled={uploading}
          onChange={(event) => upload(event.target.files?.[0])}
        />
      </label>
      <p className="upload-help">PDF only · up to the locally configured upload limit. Indexing continues in the background.</p>

      <div className="worker-row" aria-live="polite">
        <span className={worker?.running ? "status-dot ok-dot" : "status-dot bad-dot"} />
        <span>
          {worker?.running
            ? worker.active_document_id
              ? `Indexing document ${worker.active_document_id}`
              : `${worker.queued_count} queued`
            : "Indexer unavailable"}
        </span>
      </div>

      <div className="lore-list">
        {lore.length === 0 && <p className="hint">No PDFs uploaded yet.</p>}
        {lore.map((document) => (
          <article className={`lore-row ${document.status}`} key={document.id}>
            <FileText size={15} aria-hidden="true" />
            <div>
              <strong title={document.filename}>{document.filename}</strong>
              <small>
                {document.status} · {formatBytes(document.size_bytes)} · {document.chunks} chunks
              </small>
              {document.error && <span className="lore-error">{document.error}</span>}
            </div>
            <button
              type="button"
              className="icon-button danger"
              disabled={document.status === "indexing"}
              onClick={() => setConfirmDeleteId(document.id)}
              aria-label={`Delete ${document.filename}`}
            >
              <Trash2 size={14} aria-hidden="true" />
            </button>
          </article>
        ))}
      </div>

      {lore.some((document) => document.status === "error") && (
        <button type="button" className="secondary small-action" onClick={() => refresh(true)}>
          Retry failed documents
        </button>
      )}
      {confirmDeleteId !== null && (() => {
        const document = lore.find((item) => item.id === confirmDeleteId);
        return <Dialog titleId="delete-lore-title" onClose={() => setConfirmDeleteId(null)} className="confirm-dialog">
          <div className="modal-head"><div><span className="eyebrow">Remove source</span><h2 id="delete-lore-title">Delete {document?.filename ?? "this lore document"}?</h2></div>
            <button type="button" className="icon-button" onClick={() => setConfirmDeleteId(null)} aria-label="Cancel deletion"><X size={17} /></button></div>
          <p className="dialog-copy">The managed PDF and its local search index will be removed. Campaign story text already generated from it remains unchanged.</p>
          <div className="modal-actions"><button type="button" className="secondary" onClick={() => setConfirmDeleteId(null)}>Cancel</button>
            <button type="button" className="danger-button" onClick={async () => { try { await onDelete(confirmDeleteId); setConfirmDeleteId(null); } catch { /* Parent reports the error. */ } }}><Trash2 size={15} />Delete lore</button></div>
        </Dialog>;
      })()}
    </section>
  );
}
