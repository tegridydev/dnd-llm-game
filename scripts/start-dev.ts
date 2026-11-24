import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { httpUrl } from "./network";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const frontend = join(root, "frontend");
const production = process.argv.includes("--production");
const isWindows = process.platform === "win32";
const apiHost = process.env.API_HOST ?? "127.0.0.1";
const apiPort = Number(process.env.API_PORT ?? "8765");
const frontendHost = process.env.FRONTEND_HOST ?? "127.0.0.1";
const frontendPort = Number(process.env.FRONTEND_PORT ?? "5173");
const localHosts = new Set(["127.0.0.1", "localhost", "::1"]);
if (![apiHost, frontendHost].every((host) => localHosts.has(host)) ||
    ![apiPort, frontendPort].every((port) => Number.isInteger(port) && port >= 1024 && port <= 65535) ||
    apiPort === frontendPort) {
  console.error("Set distinct local API and frontend ports on 127.0.0.1, localhost or ::1.");
  process.exit(1);
}
const apiUrl = httpUrl(apiHost, apiPort);
const webUrl = httpUrl(frontendHost, frontendPort);
const venvPython = isWindows
  ? join(root, ".venv", "Scripts", "python.exe")
  : join(root, ".venv", "bin", "python");

function spawnSync(command: string[], cwd = root, env: Record<string, string> = {}) {
  const child = Bun.spawnSync({
    cmd: command,
    cwd,
    env: { ...process.env, ...env },
    stdout: "inherit",
    stderr: "inherit"
  });
  if (child.exitCode !== 0) process.exit(child.exitCode ?? 1);
}

function canRun(command: string[]): boolean {
  const child = Bun.spawnSync({ cmd: command, stdout: "pipe", stderr: "pipe" });
  return child.exitCode === 0;
}

async function waitFor(url: string, label: string, timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs;
  let lastError = "not reachable";
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(4000) });
      if (response.ok) return;
      lastError = `HTTP ${response.status}`;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
    await Bun.sleep(250);
  }
  throw new Error(`${label} did not become ready: ${lastError}`);
}

async function openBrowser(url: string) {
  const commands =
    process.platform === "darwin"
      ? [["open", url]]
      : isWindows
        ? [["cmd", "/c", "start", "", url]]
        : [["xdg-open", url]];
  for (const command of commands) {
    try {
      Bun.spawn({ cmd: command, stdout: "ignore", stderr: "ignore" });
      return;
    } catch {
      // Browser launch is optional; the URL is printed below.
    }
  }
}

if (!canRun(["uv", "--version"])) {
  console.error("uv is required. Install it explicitly from https://docs.astral.sh/uv/.");
  process.exit(1);
}

console.log(`Mode: ${production ? "production preview" : "development"}`);
console.log("Synchronising the Python environment...");
spawnSync(["uv", "sync", "--locked", ...(production ? ["--no-dev"] : ["--extra", "dev"])]);
console.log("Installing frontend packages from bun.lock...");
spawnSync(["bun", "install", "--frozen-lockfile"], frontend);
if (production) {
  console.log("Building the frontend...");
  spawnSync(["bun", "run", "build"], frontend);
}

const backendEnvironment = {
  PYTHONPATH: join(root, "backend"),
  API_HOST: apiHost,
  API_PORT: String(apiPort),
  API_RELOAD: production || isWindows ? "false" : "true",
  FRONTEND_HOST: frontendHost,
  FRONTEND_PORT: String(frontendPort)
};

const apiProcess = Bun.spawn({
  cmd: [venvPython, "-m", "dndllm26.main"],
  cwd: root,
  env: { ...process.env, ...backendEnvironment },
  stdout: "inherit",
  stderr: "inherit"
});

const webEnvironment = Object.fromEntries(
  Object.entries(process.env).filter(([key]) => !/^(LLM_|OLLAMA_)/i.test(key))
);
const webProcess = Bun.spawn({
  cmd: production
    ? ["bun", "run", "preview", "--", "--host", frontendHost, "--port", String(frontendPort), "--strictPort"]
    : ["bun", "run", "dev", "--", "--host", frontendHost, "--port", String(frontendPort), "--strictPort"],
  cwd: frontend,
  env: { ...webEnvironment, VITE_API_BASE: `${apiUrl}/api` },
  stdout: "inherit",
  stderr: "inherit"
});

function shutdown() {
  apiProcess.kill();
  webProcess.kill();
}

process.on("SIGINT", () => {
  shutdown();
  process.exit(130);
});
process.on("SIGTERM", () => {
  shutdown();
  process.exit(143);
});

try {
  await Promise.all([
    waitFor(`${apiUrl}/api/health/ready`, "API"),
    waitFor(webUrl, "frontend")
  ]);
  console.log(`API: ${apiUrl}`);
  console.log(`Web: ${webUrl}`);
  if (!process.argv.includes("--no-browser")) await openBrowser(webUrl);
  const exitCode = await Promise.race([apiProcess.exited, webProcess.exited]);
  shutdown();
  process.exit(exitCode ?? 0);
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  shutdown();
  process.exit(1);
}
