import { access, copyFile, mkdir, readFile, rm } from "node:fs/promises";
import { relative, resolve, sep } from "node:path";

const output = resolve(import.meta.dir, "dist");
await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
const result = await Bun.build({
  entrypoints: [resolve(import.meta.dir, "src/main.ts")],
  outdir: output,
  target: "browser",
  minify: true,
});
if (!result.success) {
  for (const error of result.logs) console.error(error);
  process.exit(1);
}
await copyFile(resolve(import.meta.dir, "src/index.html"), resolve(output, "index.html"));
await copyFile(resolve(import.meta.dir, "src/style.css"), resolve(output, "style.css"));

const html = await readFile(resolve(output, "index.html"), "utf8");
const references = [...html.matchAll(/\b(?:src|href)\s*=\s*["']([^"']+)["']/gi)]
  .map((match) => match[1])
  .filter((reference) => !/^(?:[a-z][a-z0-9+.-]*:|\/\/|#)/i.test(reference));
for (const reference of references) {
  const assetPath = resolve(output, decodeURIComponent(reference.split(/[?#]/, 1)[0]));
  const fromOutput = relative(output, assetPath);
  if (fromOutput === ".." || fromOutput.startsWith(`..${sep}`) || fromOutput.startsWith(sep)) {
    throw new Error(`UI asset reference escapes the output directory: ${reference}`);
  }
  try {
    await access(assetPath);
  } catch {
    throw new Error(`UI asset reference does not exist in the output: ${reference}`);
  }
}
