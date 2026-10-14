import fs from "fs/promises";
import os from "os";
import path from "path";
import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import * as acp from "@agentclientprotocol/sdk";
import type { Config } from "../config.ts";
import type { ModelData } from "../compilers/run.ts";
import {
  compilerUsage,
  type Compiler,
  type CompilerError,
  type CompilerParams,
  type CompilerResult,
  type CompilerSuccessData,
} from "../libocto/compilers/compiler-interface.ts";
import type { Agent, AssistantMessage } from "../libocto/llm-ir.ts";
import type { LoadedTools, ToolCall } from "../libocto/tool-def.ts";
import { err, ok, type Result } from "../libocto/result.ts";
import type { octoAgent } from "../ir/octo-ir.ts";
import toolMap from "../tools/tool-defs/index.ts";
import { fileExists } from "../fs-utils.ts";
import { loadSession } from "../session-history/index.ts";
import { runAcpServer, type AcpServer } from "./server.ts";
import { promptToUserContent } from "./content.ts";

// Contract tests: a real ACP client talks to runAcpServer over in-memory pipes, with a scripted
// compiler in place of the model.

type OctoAssistant = AssistantMessage<(typeof octoAgent)["tools"]>;
type CreateCall = Extract<ToolCall<typeof toolMap>, { name: "create" }>;
type CompilerOutcome = Result<CompilerSuccessData<typeof octoAgent>, CompilerError>;
type ScriptStep = (args: {
  onTokens: (tokens: string, type: "reasoning" | "content") => void;
  abortSignal: AbortSignal;
  irsJson: string;
}) => CompilerOutcome | Promise<CompilerOutcome>;

const API_KEY_ENV = "OCTO_ACP_TEST_API_KEY";
const config: Config = {
  yourName: "Test",
  models: [
    {
      nickname: "test-model",
      model: "test-model",
      context: 128_000,
      baseUrl: "http://localhost",
      apiEnvVar: API_KEY_ENV,
    },
  ],
};

function reply(opts: {
  content?: string;
  toolCalls?: OctoAssistant["toolCalls"];
}): CompilerOutcome {
  const output: OctoAssistant = {
    role: "assistant",
    content: opts.content ?? "",
    usage: compilerUsage(1, 1),
    toolCalls: opts.toolCalls,
  };
  return ok({ output, curl: "curl", headers: new Headers(), usage: output.usage });
}

function streamed(chunks: string[], reasoning: string[] = []): ScriptStep {
  return ({ onTokens }) => {
    for (const r of reasoning) onTokens(r, "reasoning");
    for (const c of chunks) onTokens(c, "content");
    return reply({ content: chunks.join("") });
  };
}

function callsTools(...toolCalls: CreateCall[]): ScriptStep {
  return () => reply({ toolCalls });
}

function createCall(id: string, filePath: string, content: string): CreateCall {
  return {
    type: "tool-call",
    name: "create",
    toolCallId: id,
    original: { filePath, content },
    parsed: { filePath, content },
  };
}

const paymentError: ScriptStep = () =>
  err({
    type: "payment-error",
    requestError: "mock: out of credits",
    curl: "curl",
    headers: new Headers(),
  });

const hangsUntilAborted: ScriptStep = async ({ onTokens, abortSignal }) => {
  onTokens("partial", "content");
  await new Promise<void>(resolve => {
    if (abortSignal.aborted) return resolve();
    abortSignal.addEventListener("abort", () => resolve(), { once: true });
  });
  return reply({ content: "unreached" });
};

function scriptedCompiler(script: (index: number, irsJson: string) => ScriptStep) {
  const calls: string[] = [];
  const runCompiler: Compiler<ModelData> = async <
    A extends Agent<any, any, any>,
    Tools extends Partial<LoadedTools<A["tools"]>> | undefined = undefined,
  >(
    params: CompilerParams<A, ModelData, Tools>,
  ): Promise<CompilerResult<A, Tools>> => {
    const irsJson = JSON.stringify(params.irs);
    const index = calls.length;
    calls.push(irsJson);
    const result = await script(
      index,
      irsJson,
    )({
      onTokens: params.onTokens as (t: string, type: "reasoning" | "content") => void,
      abortSignal: params.abortSignal,
      irsJson,
    });
    return result as typeof result & CompilerResult<A, Tools>;
  };
  return { runCompiler, calls };
}

// When the script runs out, return a payment error instead of throwing.
function queued(steps: ScriptStep[]) {
  return scriptedCompiler(index => steps[index] ?? paymentError);
}

type PermissionHandler = (
  req: acp.RequestPermissionRequest,
) => Promise<acp.RequestPermissionResponse>;

function choose(kind: acp.PermissionOptionKind): PermissionHandler {
  return async req => {
    const option = req.options.find(o => o.kind === kind);
    if (option == null) throw new Error(`no ${kind} option offered`);
    return { outcome: { outcome: "selected", optionId: option.optionId } };
  };
}

type Harness = {
  agent: acp.ClientContext;
  server: AcpServer;
  updates: acp.SessionNotification[];
  permissionRequests: acp.RequestPermissionRequest[];
  rawLines: string[];
  permission: PermissionHandler;
  close: () => Promise<void>;
};

const harnesses: Harness[] = [];
const tempDirs: string[] = [];
const envBefore = process.env[API_KEY_ENV];

beforeEach(() => {
  process.env[API_KEY_ENV] = "test-key";
});

afterEach(async () => {
  while (harnesses.length > 0) await harnesses.pop()!.close();
  while (tempDirs.length > 0) await fs.rm(tempDirs.pop()!, { recursive: true, force: true });
  if (envBefore == null) delete process.env[API_KEY_ENV];
  else process.env[API_KEY_ENV] = envBefore;
});

async function tempCwd(): Promise<string> {
  const dir = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), "octo-acp-")));
  tempDirs.push(dir);
  return dir;
}

async function collectLines(reader: ReadableStreamDefaultReader<Uint8Array>, into: string[]) {
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop()!;
    into.push(...lines.filter(line => line !== ""));
  }
  if (buffer !== "") into.push(buffer);
}

async function startHarness(runCompiler: Compiler<ModelData>): Promise<Harness> {
  const toAgent = new TransformStream<Uint8Array, Uint8Array>();
  const toClient = new TransformStream<Uint8Array, Uint8Array>();
  const [clientInput, rawOutput] = toClient.readable.tee();

  const server = runAcpServer({
    config,
    unchained: false,
    input: toAgent.readable,
    output: toClient.writable,
    runCompiler,
  });

  const harness = {
    server,
    updates: [],
    permissionRequests: [],
    rawLines: [],
    permission: choose("allow_once"),
  } as unknown as Harness;
  // The SDK never closes its output stream, so the raw tap is cancelled on close instead.
  const rawReader = rawOutput.getReader();
  const rawDone = collectLines(rawReader, harness.rawLines);

  const connection = acp
    .client({ name: "acp-test" })
    .onRequest("session/request_permission", async ({ params }) => {
      harness.permissionRequests.push(params);
      return await harness.permission(params);
    })
    .onNotification("session/update", ({ params }) => {
      harness.updates.push(params);
    })
    .connect(acp.ndJsonStream(toAgent.writable, clientInput));

  harness.agent = connection.agent;
  harness.close = async () => {
    await server.close();
    connection.close();
    await server.closed;
    await rawReader.cancel();
    await rawDone;
  };
  harnesses.push(harness);

  await harness.agent.request("initialize", {
    protocolVersion: acp.PROTOCOL_VERSION,
    clientCapabilities: {},
  });
  return harness;
}

async function newSession(h: Harness, cwd: string): Promise<string> {
  const { sessionId } = await h.agent.request("session/new", { cwd, mcpServers: [] });
  return sessionId;
}

function prompt(h: Harness, sessionId: string, text: string) {
  return h.agent.request("session/prompt", {
    sessionId,
    prompt: [{ type: "text", text }],
  });
}

function updatesFor(h: Harness, sessionId: string): acp.SessionUpdate[] {
  return h.updates.filter(u => u.sessionId === sessionId).map(u => u.update);
}

function messageText(h: Harness, sessionId: string): string {
  return updatesFor(h, sessionId)
    .flatMap(u =>
      u.sessionUpdate === "agent_message_chunk" && u.content.type === "text"
        ? [u.content.text]
        : [],
    )
    .join("");
}

function toolStatuses(h: Harness, sessionId: string, toolCallId: string): string[] {
  return updatesFor(h, sessionId).flatMap(u =>
    (u.sessionUpdate === "tool_call" || u.sessionUpdate === "tool_call_update") &&
    u.toolCallId === toolCallId &&
    u.status != null
      ? [u.status]
      : [],
  );
}

function historyRoles(sessionId: string): string[] {
  const loaded = loadSession(sessionId);
  if (loaded == null) throw new Error(`session ${sessionId} was not persisted`);
  return loaded.history.flatMap(node => (node.type === "llm-ir" ? [node.ir.role] : []));
}

async function waitFor<T>(cond: () => T | null | undefined | false, timeoutMs = 10_000) {
  const start = Date.now();
  while (true) {
    const value = cond();
    if (value != null && value !== false) return value;
    if (Date.now() - start > timeoutMs) throw new Error("Timed out waiting for condition");
    await new Promise(resolve => setTimeout(resolve, 10));
  }
}

describe("octo acp", () => {
  it("initializes with protocol v1 and no session loading", async () => {
    const h = await startHarness(queued([]).runCompiler);
    const response = await h.agent.request("initialize", {
      protocolVersion: acp.PROTOCOL_VERSION,
      clientCapabilities: {},
    });
    expect(response.protocolVersion).toBe(1);
    expect(response.agentCapabilities?.loadSession).toBe(false);
    expect(response.authMethods).toEqual([]);
  });

  it("streams a text turn and ends it with end_turn", async () => {
    const h = await startHarness(
      queued([streamed(["Hel", "lo, ", "Crawler"], ["thinking..."])]).runCompiler,
    );
    const sessionId = await newSession(h, await tempCwd());

    const response = await prompt(h, sessionId, "hi");

    expect(response.stopReason).toBe("end_turn");
    expect(messageText(h, sessionId)).toBe("Hello, Crawler");
    expect(updatesFor(h, sessionId)).toContainEqual({
      sessionUpdate: "agent_thought_chunk",
      content: { type: "text", text: "thinking..." },
    });
    expect(historyRoles(sessionId)).toEqual(["user", "assistant"]);
  });

  it("asks permission for a tool, runs it in the session cwd once allowed", async () => {
    const h = await startHarness(
      queued([callsTools(createCall("t1", "notes.txt", "hello")), streamed(["done"])]).runCompiler,
    );
    const cwd = await tempCwd();
    const sessionId = await newSession(h, cwd);

    const response = await prompt(h, sessionId, "write notes");

    expect(response.stopReason).toBe("end_turn");
    const announced = updatesFor(h, sessionId).find(u => u.sessionUpdate === "tool_call");
    expect(announced).toMatchObject({
      sessionUpdate: "tool_call",
      toolCallId: "t1",
      kind: "edit",
      status: "pending",
      locations: [{ path: path.join(cwd, "notes.txt") }],
    });
    expect(h.permissionRequests).toHaveLength(1);
    expect(h.permissionRequests[0].toolCall.toolCallId).toBe("t1");
    expect(toolStatuses(h, sessionId, "t1")).toEqual(["pending", "in_progress", "completed"]);
    expect(await fs.readFile(path.join(cwd, "notes.txt"), "utf8")).toBe("hello");
    expect(messageText(h, sessionId)).toBe("done");
  });

  it("ends the turn on reject and steers with the next prompt", async () => {
    const compiler = queued([
      callsTools(createCall("t1", "notes.txt", "hello")),
      streamed(["steered"]),
    ]);
    const h = await startHarness(compiler.runCompiler);
    h.permission = choose("reject_once");
    const cwd = await tempCwd();
    const sessionId = await newSession(h, cwd);

    const rejected = await prompt(h, sessionId, "write notes");

    expect(rejected.stopReason).toBe("end_turn");
    expect(toolStatuses(h, sessionId, "t1")).toEqual(["pending", "failed"]);
    expect(await fileExists(path.join(cwd, "notes.txt"))).toBe(false);
    expect(compiler.calls).toHaveLength(1);

    const steered = await prompt(h, sessionId, "use a different file name");

    expect(steered.stopReason).toBe("end_turn");
    expect(compiler.calls).toHaveLength(2);
    expect(compiler.calls[1]).toContain("use a different file name");
    expect(historyRoles(sessionId)).toEqual([
      "user",
      "assistant",
      "tool-reject",
      "user",
      "assistant",
    ]);
    expect(messageText(h, sessionId)).toBe("steered");
  });

  it("stops asking for the same kind of tool after allow_always", async () => {
    const h = await startHarness(
      queued([
        callsTools(createCall("t1", "a.txt", "a")),
        callsTools(createCall("t2", "b.txt", "b")),
        streamed(["done"]),
      ]).runCompiler,
    );
    h.permission = choose("allow_always");
    const cwd = await tempCwd();
    const sessionId = await newSession(h, cwd);

    const response = await prompt(h, sessionId, "write two files");

    expect(response.stopReason).toBe("end_turn");
    expect(h.permissionRequests.map(r => r.toolCall.toolCallId)).toEqual(["t1"]);
    expect(await fs.readFile(path.join(cwd, "a.txt"), "utf8")).toBe("a");
    expect(await fs.readFile(path.join(cwd, "b.txt"), "utf8")).toBe("b");
  });

  it("cancels a streaming turn with stopReason cancelled", async () => {
    const h = await startHarness(queued([hangsUntilAborted, streamed(["after"])]).runCompiler);
    const sessionId = await newSession(h, await tempCwd());

    const turn = prompt(h, sessionId, "go");
    await waitFor(() => messageText(h, sessionId) === "partial");
    await h.agent.notify("session/cancel", { sessionId });

    expect((await turn).stopReason).toBe("cancelled");
    expect((await prompt(h, sessionId, "again")).stopReason).toBe("end_turn");
  });

  it("cancels a turn waiting on permission and records the skipped tool", async () => {
    const h = await startHarness(
      queued([callsTools(createCall("t1", "notes.txt", "hello"))]).runCompiler,
    );
    let answer!: (r: acp.RequestPermissionResponse) => void;
    h.permission = () => new Promise(resolve => (answer = resolve));
    const cwd = await tempCwd();
    const sessionId = await newSession(h, cwd);

    const turn = prompt(h, sessionId, "write notes");
    await waitFor(() => h.permissionRequests.length === 1);
    await h.agent.notify("session/cancel", { sessionId });
    answer({ outcome: { outcome: "cancelled" } });

    expect((await turn).stopReason).toBe("cancelled");
    expect(await fileExists(path.join(cwd, "notes.txt"))).toBe(false);
    expect(historyRoles(sessionId)).toEqual(["user", "assistant", "tool-skip-output"]);
    expect(toolStatuses(h, sessionId, "t1")).toEqual(["pending", "failed"]);
  });

  it("fails a turn that ends in an error and recovers on the next prompt", async () => {
    const compiler = queued([paymentError, streamed(["retried"]), streamed([" answered"])]);
    const h = await startHarness(compiler.runCompiler);
    const sessionId = await newSession(h, await tempCwd());

    const failed = prompt(h, sessionId, "first");
    await expect(failed).rejects.toThrow("out of credits");

    const recovered = await prompt(h, sessionId, "second");

    expect(recovered.stopReason).toBe("end_turn");
    expect(compiler.calls).toHaveLength(3);
    expect(compiler.calls[2]).toContain("second");
    expect(messageText(h, sessionId)).toBe("retried answered");
  });

  it("runs two sessions on one connection independently, each in its own cwd", async () => {
    // Each session's model writes a file named after its own prompt, then finishes.
    const compiler = scriptedCompiler((_, irsJson) => {
      const who = irsJson.includes("session-A") ? "A" : "B";
      if (irsJson.includes("tool-output") || irsJson.includes("file-mutate")) {
        return streamed([`done ${who}`]);
      }
      return callsTools(createCall(`call-${who}`, "who.txt", who));
    });
    const h = await startHarness(compiler.runCompiler);
    const [cwdA, cwdB] = [await tempCwd(), await tempCwd()];
    const [a, b] = [await newSession(h, cwdA), await newSession(h, cwdB)];
    expect(a).not.toBe(b);

    const [ra, rb] = await Promise.all([prompt(h, a, "session-A"), prompt(h, b, "session-B")]);

    expect([ra.stopReason, rb.stopReason]).toEqual(["end_turn", "end_turn"]);
    expect(await fs.readFile(path.join(cwdA, "who.txt"), "utf8")).toBe("A");
    expect(await fs.readFile(path.join(cwdB, "who.txt"), "utf8")).toBe("B");
    expect(messageText(h, a)).toBe("done A");
    expect(messageText(h, b)).toBe("done B");
  });

  it("writes nothing but JSON-RPC messages to its output", async () => {
    const h = await startHarness(
      queued([callsTools(createCall("t1", "notes.txt", "hi")), streamed(["ok"])]).runCompiler,
    );
    const sessionId = await newSession(h, await tempCwd());
    await prompt(h, sessionId, "go");
    await h.close();

    expect(h.rawLines.length).toBeGreaterThan(0);
    for (const line of h.rawLines) {
      expect(JSON.parse(line)).toMatchObject({ jsonrpc: "2.0" });
    }
  });

  it("rejects relative cwds and unsupported prompt content", async () => {
    const h = await startHarness(queued([]).runCompiler);
    await expect(
      h.agent.request("session/new", { cwd: "relative", mcpServers: [] }),
    ).rejects.toThrow();
    const sessionId = await newSession(h, await tempCwd());
    await expect(
      h.agent.request("session/prompt", {
        sessionId,
        prompt: [{ type: "image", data: "", mimeType: "image/png" }],
      }),
    ).rejects.toThrow("image");
  });
});

function mulberry32(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomText(rand: () => number): string {
  const alphabet = "ab \n\t{}\"'😀é/\\";
  const length = Math.floor(rand() * 12);
  let text = "";
  for (let i = 0; i < length; i++) text += alphabet[Math.floor(rand() * alphabet.length)];
  return text;
}

describe("acp content conversion", () => {
  it("preserves every text block, in order, alongside resource links", () => {
    const rand = mulberry32(0xac9);
    for (let run = 0; run < 500; run++) {
      const blocks: acp.ContentBlock[] = Array.from({ length: Math.floor(rand() * 8) }, () =>
        rand() < 0.7
          ? { type: "text", text: randomText(rand) }
          : {
              type: "resource_link",
              name: `f${Math.floor(rand() * 9)}.ts`,
              uri: `file:///tmp/dir/f${Math.floor(rand() * 9)}.ts`,
            },
      );

      const result = promptToUserContent(blocks);

      expect(result.success).toBe(true);
      if (!result.success) continue;
      expect(result.data).toHaveLength(blocks.length);
      blocks.forEach((block, i) => {
        const part = result.data[i];
        if (part.type !== "text") throw new Error("expected text content");
        if (block.type === "text") expect(part.content).toBe(block.text);
        if (block.type === "resource_link")
          expect(part.content).toContain(new URL(block.uri).pathname);
      });
    }
  });

  it("still mentions file links that name no local path, instead of failing", () => {
    for (const uri of ["file://host/share/a.ts", "file:///tmp/%ZZ.ts"]) {
      const result = promptToUserContent([{ type: "resource_link", name: "", uri }]);
      expect(result.success).toBe(true);
      if (!result.success) continue;
      const [part] = result.data;
      expect(part.type === "text" && part.content).toEndWith(uri.slice(uri.lastIndexOf("/")));
    }
  });

  it("refuses any prompt containing non-text media", () => {
    const rand = mulberry32(7);
    for (let run = 0; run < 200; run++) {
      const blocks: acp.ContentBlock[] = Array.from({ length: Math.floor(rand() * 5) }, () => ({
        type: "text" as const,
        text: randomText(rand),
      }));
      blocks.splice(Math.floor(rand() * (blocks.length + 1)), 0, {
        type: "image",
        data: "",
        mimeType: "image/png",
      });
      expect(promptToUserContent(blocks).success).toBe(false);
    }
  });
});
