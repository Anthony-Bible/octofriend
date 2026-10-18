import path from "path";
import * as acp from "@agentclientprotocol/sdk";
import { APP_METADATA, Config } from "../config.ts";
import { run, type ModelData } from "../compilers/run.ts";
import type { Compiler } from "../libocto/compilers/compiler-interface.ts";
import { AcpSession } from "./session.ts";
import { promptToUserContent } from "./content.ts";

export type AcpServerArgs = {
  config: Config;
  configPath?: string;
  unchained: boolean;
  input: ReadableStream<Uint8Array>;
  output: WritableStream<Uint8Array>;
  runCompiler?: Compiler<ModelData>;
};

export type AcpServer = {
  // Resolves once the client disconnects and every session has shut down.
  closed: Promise<void>;
  close: () => Promise<void>;
};

// Each session/new gets its own Trajectory, so one connection can run several sessions.
export function runAcpServer(args: AcpServerArgs): AcpServer {
  const sessions = new Map<string, AcpSession>();
  const runCompiler = args.runCompiler ?? run;
  let client: acp.AgentContext | null = null;

  const getSession = (sessionId: string) => {
    const session = sessions.get(sessionId);
    if (session == null) throw acp.RequestError.invalidParams(undefined, "Unknown session");
    return session;
  };

  const connection = acp
    .agent({ name: "octofriend" })
    .onConnect(conn => {
      client = conn.client;
    })
    .onRequest("initialize", () => ({
      protocolVersion: acp.PROTOCOL_VERSION,
      agentCapabilities: { loadSession: false, promptCapabilities: {} },
      agentInfo: { name: "octofriend", title: "Octo", version: APP_METADATA.version },
      authMethods: [],
    }))
    .onRequest("session/new", ({ params }) => {
      if (!path.isAbsolute(params.cwd)) {
        throw acp.RequestError.invalidParams(undefined, "cwd must be an absolute path");
      }
      if (client == null) {
        throw acp.RequestError.internalError(undefined, "Connection is not established");
      }
      const sessionId = crypto.randomUUID();
      sessions.set(
        sessionId,
        new AcpSession({
          sessionId,
          cwd: params.cwd,
          config: args.config,
          configPath: args.configPath,
          unchained: args.unchained,
          client,
          runCompiler,
          // Aborted sessions are terminal: stop routing to them instead of
          // accumulating dead entries for the life of the connection.
          onEnd: () => {
            sessions.delete(sessionId);
          },
        }),
      );
      return { sessionId };
    })
    .onRequest("session/prompt", async ({ params }) => {
      const session = getSession(params.sessionId);
      const content = promptToUserContent(params.prompt);
      if (!content.success) throw acp.RequestError.invalidParams(undefined, content.error);
      return await session.prompt(content.data);
    })
    .onNotification("session/cancel", async ({ params }) => {
      await sessions.get(params.sessionId)?.cancel();
    })
    .connect(acp.ndJsonStream(args.output, args.input));

  let shutdown: Promise<void> | null = null;
  const shutdownSessions = () => {
    shutdown ??= Promise.all([...sessions.values()].map(s => s.shutdown())).then(() => {});
    return shutdown;
  };

  return {
    closed: connection.closed.then(shutdownSessions),
    close: async () => {
      await shutdownSessions();
      connection.close();
    },
  };
}
