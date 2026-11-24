import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";

type Chunk = { file: string; isEntry?: boolean; imports?: string[] };
const dist = resolve(import.meta.dir, "../frontend/dist");
const manifest = JSON.parse(readFileSync(join(dist, ".vite", "manifest.json"), "utf8")) as Record<string, Chunk>;
const entries = Object.keys(manifest).filter((key) => manifest[key].isEntry);
if (!entries.length) throw new Error("The Vite manifest contains no application entry.");

const initial = new Set<string>();
function addInitial(key: string): void {
  const chunk = manifest[key];
  if (!chunk) throw new Error(`Missing bundle manifest entry ${key}`);
  if (initial.has(chunk.file)) return;
  initial.add(chunk.file);
  for (const imported of chunk.imports ?? []) addInitial(imported);
}
entries.forEach(addInitial);

const weight = (paths: Iterable<string>) => [...paths]
  .filter((name) => name.endsWith(".js"))
  .reduce((total, name) => total + statSync(join(dist, name)).size, 0);
const all = readdirSync(join(dist, "assets")).map((name) => `assets/${name}`);
for (const [label, bytes, budget] of [
  ["Initial JavaScript", weight(initial), 276_000],
  ["JavaScript including optional panels", weight(all), 300_000]
] as const) {
  if (bytes > budget) throw new Error(`${label}: ${bytes} bytes exceeds the ${budget} byte budget.`);
  console.log(`${label}: ${bytes}/${budget} bytes.`);
}
