import { expect, test } from "bun:test"
import { readFile } from "node:fs/promises"

test("relinking instructions identify the locked OpenTUI release", async () => {
  const root = new URL("../", import.meta.url)
  const manifest = JSON.parse(await readFile(new URL("package.json", root), "utf8"))
  const version: string = manifest.dependencies["@opentui/core"]
  const instructions = await readFile(new URL("RELINKING.md", root), "utf8")
  expect(version).toMatch(/^\d+\.\d+\.\d+$/)
  expect(instructions).toContain(`OpenTUI ${version}`)
  expect(instructions).toContain(`https://www.npmjs.com/package/@opentui/core/v/${version}`)
})
