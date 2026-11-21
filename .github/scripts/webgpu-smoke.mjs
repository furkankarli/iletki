import { spawn } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";

const directory = process.argv[2] ?? "target/webgpu-smoke";
const browserBinary = process.env.CHROME_EXECUTABLE ?? "google-chrome";
const assets = new Map([
  ["/smoke.js", ["text/javascript", await readFile(join(directory, "smoke.js"))]],
  ["/smoke_bg.wasm", ["application/wasm", await readFile(join(directory, "smoke_bg.wasm"))]],
]);
const page = `<!doctype html><title>iletki WebGPU smoke</title>
<script type="module">
  let result;
  try {
    const { default: init } = await import("/smoke.js");
    await init();
    if (!globalThis.iletkiWebGpuSmoke) throw new Error("Missing WebGPU test");
    result = { report: await globalThis.iletkiWebGpuSmoke };
  } catch (error) {
    result = { error: String(error) };
  }
  await fetch("/result", { method: "POST", body: JSON.stringify(result) });
</script>`;
assets.set("/", ["text/html", page]);

const profile = await mkdtemp(join(tmpdir(), "iletki-webgpu-"));
let browser;
let timeout;
let diagnostics = "";
const { promise: outcome, resolve, reject } = Promise.withResolvers();
const server = createServer(async (request, response) => {
  if (request.method === "POST" && request.url === "/result") {
    try {
      let body = "";
      for await (const chunk of request) body += chunk;
      const { report, error } = JSON.parse(body);
      response.writeHead(204).end();
      if (typeof report === "string" && report.startsWith("BrowserWebGpu ")) resolve(report);
      else reject(new Error(error ?? "Missing BrowserWebGpu result"));
    } catch (error) {
      response.writeHead(400).end();
      reject(error);
    }
    return;
  }
  const asset = request.method === "GET" && assets.get(request.url);
  if (asset) {
    const [type, body] = asset;
    response.writeHead(200, { "Content-Type": type }).end(body);
  } else {
    response.writeHead(404).end();
  }
});

try {
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  browser = spawn(browserBinary, [
    "--headless=new",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-networking",
    "--enable-unsafe-webgpu",
    "--enable-unsafe-swiftshader",
    "--use-webgpu-adapter=swiftshader",
    "--use-vulkan=swiftshader",
    `--user-data-dir=${profile}`,
    `http://127.0.0.1:${server.address().port}/`,
  ], { stdio: ["ignore", "ignore", "pipe"] });
  browser.stderr.setEncoding("utf8").on("data", (chunk) => {
    diagnostics = (diagnostics + chunk).slice(-8000);
  });
  browser.on("error", reject);
  browser.on("exit", (code, signal) => reject(new Error(`Chrome exited (${code ?? signal})`)));
  timeout = setTimeout(() => reject(new Error("Timed out waiting for WebGPU render and picking")), 60_000);
  console.log(await outcome);
} catch (error) {
  if (diagnostics) console.error(diagnostics);
  console.error(error);
  process.exitCode = 1;
} finally {
  clearTimeout(timeout);
  if (browser?.pid && browser.exitCode === null && browser.signalCode === null) {
    const closed = once(browser, "close");
    browser.kill();
    await closed;
  }
  server.closeAllConnections();
  server.close();
  await rm(profile, { recursive: true, force: true, maxRetries: 3 });
}
