import { Config, ModelConfig, readAuthForModel } from "./config.ts";
import { Transport } from "./transports/transport-common.ts";
import type { ModelData } from "./compilers/run.ts";
import type { Compiler } from "./libocto/compilers/compiler-interface.ts";
import { lowerOcto } from "./compilers/lower-octo.ts";
import { autofixEdit, makeAutofixJson } from "./compilers/autofix.ts";
import { systemPrompt } from "./prompts/system-prompt.ts";
import { messageText } from "./libocto/llm-ir.ts";
import { err, ok, type Result } from "./libocto/result.ts";
import {
  DEFAULT_MAX_TOOL_OUTPUT_FRACTION,
  type TrajectoryHandler,
  type TrajectoryModelError,
  type TrajectoryParams,
} from "./libocto/trajectory.ts";
import type { PermissionGate } from "./libocto/permissions.ts";
import { loadTools } from "./tools/index.ts";
import { estimateTokens } from "./ir/count-ir-tokens.ts";
import { octoAgent, type OctoIR } from "./ir/octo-ir.ts";

export const MAX_RETRY_COUNT = 20;

const PREVIEW_CHARS = 200;

export type OctoTrajectoryArgs = {
  // Used for tool data and autofix; fixed for the trajectory's lifetime.
  config: Config;
  // Read on every request, so config changes apply to the next one.
  liveConfig: () => Config;
  model: () => ModelConfig;
  transport: Transport;
  messages: OctoIR[];
  runCompiler: Compiler<ModelData>;
  abortSignal: AbortSignal;
  permission: PermissionGate<typeof octoAgent>;
  handler: TrajectoryHandler<typeof octoAgent>;
};

// Trajectory setup shared by the TUI and the ACP server. Callers supply only the permission gate
// and event handler.
export function buildOctoTrajectoryParams(
  args: OctoTrajectoryArgs,
): TrajectoryParams<typeof octoAgent, ModelData> {
  const { config, liveConfig, model: currentModel, transport } = args;
  return {
    agent: octoAgent,
    messages: args.messages,
    abortSignal: args.abortSignal,
    systemPrompt: signal => systemPrompt({ config: liveConfig(), transport, signal }),
    model: async (): Promise<
      Result<{ model: ModelData; contextWindow: number }, TrajectoryModelError>
    > => {
      const model = currentModel();
      if (model.type === "codex") {
        const authResult = await readAuthForModel(model, liveConfig());
        if (!authResult.ok) {
          return err({ type: "auth-error", authError: authResult.error.message });
        }
        return ok({
          model: { type: "codex", auth: authResult.auth, model },
          contextWindow: model.context,
        });
      }
      const authResult = await readAuthForModel(model, liveConfig());
      if (!authResult.ok) {
        return err({ type: "auth-error", authError: authResult.error.message });
      }
      return ok({
        model: { type: "api", auth: authResult.auth, model },
        contextWindow: model.context,
      });
    },
    loadTools: signal => loadTools(transport, signal, liveConfig()),
    countTokens: irs => irs.reduce((tokens, ir) => tokens + estimateTokens(messageText(ir)), 0),
    toolContentTooLargeError: async ir => {
      let text = "";
      if (ir.role === "tool-output") {
        for (const part of ir.content) {
          if (part.type === "text") text += part.content;
        }
      } else if (ir.role === "file-read" || ir.role === "file-mutate") {
        text = ir.content;
      }
      const model = currentModel();
      const tokens = estimateTokens(text);
      const preview = text.slice(0, PREVIEW_CHARS);
      return (
        `Tool output was too large: approximately ${tokens} tokens, which is ` +
        `${DEFAULT_MAX_TOOL_OUTPUT_FRACTION * 100}% or more of the model's ` +
        `${model.context}-token context window. The output was discarded to protect the ` +
        `context window. Retry with a more targeted approach: page through the file with ` +
        `partial-read using offset/limit, narrow searches with tighter patterns or ` +
        `maxResults, or limit shell output (e.g. pipe through head/tail/grep). Before it ` +
        `was discarded, the first ${PREVIEW_CHARS} characters were preserved so you can ` +
        `inspect them; here they are:\n${preview}`
      );
    },
    toolData: config,
    runCompiler: args.runCompiler,
    lowerMessages: messages => lowerOcto(messages, currentModel().modalities),
    transport,
    errorCorrection: {
      json: makeAutofixJson(config),
      tools: {
        edit: async ({ toolCall, abortSignal: fixSignal, transport: fixTransport }) => {
          const file = await fixTransport.readFile(fixSignal, toolCall.parsed.filePath);
          const fix = await autofixEdit(config, file, toolCall.parsed, fixSignal);
          if (fix == null) return null;
          return { ...toolCall.parsed, ...fix };
        },
      },
    },
    requestErrorRetries: {
      maxRetryCount: MAX_RETRY_COUNT,
      backoffMs: 2000,
      maxBackoffMs: 30_000,
    },
    permission: args.permission,
    handler: args.handler,
  };
}
