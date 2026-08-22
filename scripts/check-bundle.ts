import { readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";

const assets = resolve(import.meta.dir, "../frontend/dist/assets");
const javascriptBytes = readdirSync(assets)
  .filter((name) => name.endsWith(".js"))
  .reduce((total, name) => total + statSync(join(assets, name)).size, 0);
const budget = 276_000;

if (javascriptBytes > budget) {
  console.error(`Frontend JavaScript is ${javascriptBytes} bytes; budget is ${budget} bytes.`);
  process.exit(1);
}
console.log(`Frontend JavaScript: ${javascriptBytes}/${budget} bytes.`);
