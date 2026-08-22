export function urlHost(host: string): string {
  const trimmed = host.trim();
  if (trimmed.startsWith("[") && trimmed.endsWith("]")) return trimmed;
  return trimmed.includes(":") ? `[${trimmed}]` : trimmed;
}

export function httpUrl(host: string, port: number): string {
  return `http://${urlHost(host)}:${port}`;
}
