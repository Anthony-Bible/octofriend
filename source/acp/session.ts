import {
  RequestError,
  type AgentContext,
  type PermissionOption,
  type PromptResponse,
  type RequestPermissionOutcome,
  type SessionUpdate,
} from "@agentclientprotocol/sdk";
import { Config, getModelFromConfig, type ModelConfig } from "../config.ts";
import type { ModelData } from "../compilers/run.ts";
import type { Compiler } from "../libocto/compilers/compiler-interface.ts";
import type { UserMessage } from "../libocto/llm-ir.ts";
import { Trajectory, type TrajectoryMode } from "../libocto/trajectory.ts";
import { buildOctoTrajectoryParams } from "../octo-trajectory.ts";
import {
  autoAllowed,
  octoPermissionGate,
  type OctoGateState,
  type OctoPermissionControl,
} from "../octo-permissions.ts";
import {
  createSession,
  insertHistoryItems,
  SessionNotFoundError,
  type Session,
} from "../session-history/index.ts";
import { serializeModelJson } from "../session-history/model-json.ts";
import { ALWAYS_REQUEST_PERMISSION_TOOLS } from "../tools/index.ts";
import { LocalTransport } from "../transports/local.ts";
import { octoAgent, type OctoIR } from "../ir/octo-ir.ts";
import { toolCallResult, toolCallStart } from "./tool-calls.ts";

type OctoMode = TrajectoryMode<typeof octoAgent>;

export type AcpSessionArgs = {
  sessionId: string;
  cwd: string;
  config: Config;
  configPath?: string;
  unchained: boolean;
  client: AgentContext;
  runCompiler: Compiler<ModelData>;
};

const ALLOW_ONCE = "allow-once";
const ALLOW_ALWAYS = "allow-always";
const REJECT_ONCE = "reject-once";

// One editor thread. A turn starts when session/prompt gives the trajectory input and ends when
// it stops: waiting for input, in an error mode, or on a rejected permission (which waits for the
// next prompt's text).
export class AcpSession {
  readonly id: string;
  readonly cwd: string;
  readonly runPromise: Promise<void>;

  private readonly client: AgentContext;
  private readonly model: ModelConfig;
  private readonly unchained: boolean;
  private readonly session: Session;
  private readonly exitController = new AbortController();
  private readonly trajectory: Trajectory<typeof octoAgent, ModelData>;
  private readonly gateState: OctoGateState = { rejectionTx: null, whitelistState: new Set() };
  // A rejection is reported to the client immediately, so the later tool-reject IR is skipped.
  private readonly reportedResults = new Set<string>();
  // Parent of the next persisted message.
  private lastNodeId: number | null = null;
  private turn: { resolve: (r: PromptResponse) => void; reject: (e: unknown) => void } | null =
    null;
  private cancelRequested = false;
  private interruptPending = false;
  // Error modes don't accept steering, so a prompt sent in one retries first and is queued once
  // the retry is running.
  private pendingSteering: UserMessage["content"] | null = null;
  private crashed: string | null = null;

  constructor(args: AcpSessionArgs) {
    this.id = args.sessionId;
    this.cwd = args.cwd;
    this.client = args.client;
    this.model = getModelFromConfig(args.config, null);
    this.unchained = args.unchained;
    this.session = createSession(
      args.cwd,
      { kind: "local", config: args.configPath, unchained: args.unchained },
      args.sessionId,
    );

    const transport = new LocalTransport(args.cwd);
    this.trajectory = new Trajectory(
      buildOctoTrajectoryParams({
        config: args.config,
        liveConfig: () => args.config,
        model: () => this.model,
        transport,
        messages: [],
        runCompiler: args.runCompiler,
        abortSignal: this.exitController.signal,
        permission: octoPermissionGate({
          state: this.gateState,
          onWhitelist: whitelistState => {
            this.gateState.whitelistState = whitelistState;
          },
          onBeginRejection: rejectionTx => {
            this.gateState.rejectionTx = rejectionTx;
          },
          onCommitRejection: () => {
            this.gateState.rejectionTx = null;
          },
          controller: control => this.decidePermission(control),
        }),
        handler: {
          modeChange: mode => this.onModeChange(mode),
          onMessage: ir => this.onMessage(ir),
          responseProgress: async ({ delta }) => {
            const sessionUpdate =
              delta.type === "content"
                ? "agent_message_chunk"
                : delta.type === "reasoning"
                  ? "agent_thought_chunk"
                  : null;
            if (sessionUpdate == null || delta.value === "") return;
            await this.update({ sessionUpdate, content: { type: "text", text: delta.value } });
          },
        },
      }),
    );

    this.runPromise = this.trajectory.run().catch(e => {
      this.crashed = e instanceof Error ? e.message : String(e);
      console.error(`ACP session ${this.id} crashed:`, e);
      this.settle(t => t.reject(RequestError.internalError(undefined, this.crashed!)));
    });
  }

  async prompt(content: UserMessage["content"]): Promise<PromptResponse> {
    if (this.turn != null) {
      throw RequestError.invalidRequest(undefined, "A prompt is already running in this session");
    }
    if (this.crashed != null) throw RequestError.internalError(undefined, this.crashed);
    const mode = this.trajectory.mode;
    if (mode.mode === "aborted") {
      throw RequestError.invalidRequest(undefined, "This session has ended");
    }

    const done = new Promise<PromptResponse>((resolve, reject) => {
      this.turn = { resolve, reject };
    });
    this.cancelRequested = false;
    this.interruptPending = false;

    const parked = this.gateState.rejectionTx;
    if (parked != null) {
      parked.commitRejection(content);
    } else if ("control" in mode && "retry" in mode.control) {
      this.pendingSteering = content;
      mode.control.retry();
    } else if ("control" in mode && "enqueueSteering" in mode.control) {
      await mode.control.enqueueSteering(content);
    } else {
      this.turn = null;
      throw RequestError.invalidRequest(undefined, `Session is busy (${mode.mode})`);
    }
    return await done;
  }

  // No-op between turns, including while a rejection is parked.
  async cancel(): Promise<void> {
    if (this.turn == null) return;
    this.cancelRequested = true;
    this.pendingSteering = null;
    const mode = this.trajectory.mode;
    if ("control" in mode && "interrupt" in mode.control) {
      await mode.control.interrupt();
    } else {
      // The trajectory hasn't picked up the prompt yet: interrupt once it can be.
      this.interruptPending = true;
    }
  }

  // Unanswered tool calls get skip markers, persisted before run() resolves.
  async shutdown(): Promise<void> {
    this.exitController.abort();
    await this.runPromise;
  }

  private settle(fn: (turn: NonNullable<AcpSession["turn"]>) => void) {
    const turn = this.turn;
    if (turn == null) return;
    this.turn = null;
    fn(turn);
  }

  private async onModeChange(mode: OctoMode): Promise<void> {
    if (this.pendingSteering != null && "control" in mode && "enqueueSteering" in mode.control) {
      const steering = this.pendingSteering;
      this.pendingSteering = null;
      await mode.control.enqueueSteering(steering);
    }
    if (this.interruptPending && "control" in mode && "interrupt" in mode.control) {
      this.interruptPending = false;
      await mode.control.interrupt();
    }

    switch (mode.mode) {
      case "running-tool":
        await this.update({
          sessionUpdate: "tool_call_update",
          toolCallId: mode.toolCall.toolCallId,
          status: "in_progress",
        });
        return;
      case "ready-for-request":
        this.settle(t =>
          t.resolve({ stopReason: this.cancelRequested ? "cancelled" : "end_turn" }),
        );
        return;
      case "aborted":
        this.settle(t => t.resolve({ stopReason: "cancelled" }));
        return;
      case "request-error":
      case "compaction-error":
      case "payment-error":
      case "rate-limit-error":
      case "auth-error": {
        this.pendingSteering = null;
        const message = mode.mode === "auth-error" ? mode.authError : mode.requestError;
        if (this.cancelRequested) {
          this.settle(t => t.resolve({ stopReason: "cancelled" }));
        } else {
          this.settle(t => t.reject(RequestError.internalError({ mode: mode.mode }, message)));
        }
        return;
      }
      default:
        return;
    }
  }

  private async onMessage(ir: OctoIR): Promise<void> {
    try {
      const inserted = insertHistoryItems(
        this.session,
        this.lastNodeId,
        [{ type: "llm-ir", ir }],
        serializeModelJson(this.model),
      );
      this.lastNodeId = inserted.at(-1)?.nodeId ?? this.lastNodeId;
    } catch (e) {
      if (!(e instanceof SessionNotFoundError)) throw e;
      console.error(`ACP session ${this.id} was deleted from history; ending it:`, e.message);
      this.exitController.abort();
      return;
    }

    if (ir.role === "assistant") {
      for (const toolCall of ir.toolCalls ?? []) {
        if (toolCall.type !== "tool-call") continue;
        await this.update({ sessionUpdate: "tool_call", ...toolCallStart(toolCall, this.cwd) });
      }
      return;
    }
    const result = toolCallResult(ir);
    if (result == null || this.reportedResults.delete(result.toolCallId)) return;
    await this.update({ sessionUpdate: "tool_call_update", ...result });
  }

  private async decidePermission(control: OctoPermissionControl): Promise<void> {
    const toolCall = control.toolCall;
    if (autoAllowed(toolCall, { unchained: this.unchained, ...this.gateState })) {
      control.allow();
      return;
    }

    const options: PermissionOption[] = [
      { kind: "allow_once", name: "Allow", optionId: ALLOW_ONCE },
    ];
    if (!ALWAYS_REQUEST_PERMISSION_TOOLS.includes(toolCall.name)) {
      options.push({ kind: "allow_always", name: "Always allow", optionId: ALLOW_ALWAYS });
    }
    options.push({ kind: "reject_once", name: "Reject", optionId: REJECT_ONCE });

    let outcome: RequestPermissionOutcome;
    try {
      ({ outcome } = await this.client.request("session/request_permission", {
        sessionId: this.id,
        toolCall: toolCallStart(toolCall, this.cwd),
        options,
      }));
    } catch (e) {
      // Without an answer the tool must not run: park it as a rejection and fail the turn.
      console.error(`ACP session ${this.id}: permission request failed:`, e);
      control.beginReject();
      this.settle(t =>
        t.reject(RequestError.internalError(undefined, "Permission request failed")),
      );
      return;
    }

    // A cancelled request means session/cancel already interrupted the turn.
    if (outcome.outcome === "cancelled") return;
    switch (outcome.optionId) {
      case ALLOW_ONCE:
        control.allow();
        return;
      case ALLOW_ALWAYS:
        control.allowAndWhitelist();
        return;
      default: {
        control.beginReject();
        this.reportedResults.add(toolCall.toolCallId);
        await this.update({
          sessionUpdate: "tool_call_update",
          toolCallId: toolCall.toolCallId,
          status: "failed",
        });
        this.settle(t => t.resolve({ stopReason: "end_turn" }));
        return;
      }
    }
  }

  private async update(update: SessionUpdate): Promise<void> {
    try {
      await this.client.notify("session/update", { sessionId: this.id, update });
    } catch (e) {
      console.error(`ACP session ${this.id}: failed to send update:`, e);
    }
  }
}
