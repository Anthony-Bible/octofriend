import path from "path";
import type { ToolCallContent, ToolCallLocation, ToolKind } from "@agentclientprotocol/sdk";
import type { ToolCallRequest } from "../octo-permissions.ts";
import type { OctoIR } from "../ir/octo-ir.ts";
import { contentText } from "../libocto/llm-ir.ts";
import type toolMap from "../tools/tool-defs/index.ts";

type OctoToolName = keyof typeof toolMap;

function toolKind(name: OctoToolName): ToolKind {
  switch (name) {
    case "read":
    case "partial-read":
    case "list":
    case "lsp-definition":
    case "lsp-references":
    case "lsp-hover":
    case "lsp-diagnostics":
    case "lsp-document-symbol":
    case "lsp-implementation":
    case "lsp-incoming-calls":
    case "lsp-outgoing-calls":
      return "read";
    case "edit":
    case "create":
    case "rewrite":
      return "edit";
    case "glob":
    case "grep":
      return "search";
    case "shell":
    case "background-process":
    case "manage-background-process":
      return "execute";
    case "fetch":
    case "web-search":
      return "fetch";
    case "skill":
    case "mcp":
      return "other";
    default:
      // A tool added without a mapping must not produce an undefined kind.
      return "other";
  }
}

// Titles show paths as the model wrote them; locations use absolute paths.
function toolCallTitle(toolCall: ToolCallRequest): string {
  if ("filePath" in toolCall.original) return `${toolCall.name} ${toolCall.original.filePath}`;
  switch (toolCall.name) {
    case "list":
      return `list ${toolCall.original.dirPath || "."}`;
    case "shell":
      return `shell: ${toolCall.original.cmd}`;
    case "background-process":
      return `background: ${toolCall.original.cmd}`;
    case "manage-background-process": {
      const target = toolCall.original.label ?? toolCall.original.id;
      return target == null
        ? `background processes: ${toolCall.original.action}`
        : `background processes: ${toolCall.original.action} ${target}`;
    }
    case "glob": {
      const pattern = toolCall.original.includeName ?? toolCall.original.includePath ?? "*";
      return `glob ${pattern}`;
    }
    case "grep":
      return `grep ${toolCall.original.pattern}`;
    case "fetch":
      return `fetch ${toolCall.original.url}`;
    case "web-search":
      return `search: ${toolCall.original.query}`;
    case "skill":
      return `skill ${toolCall.original.skillName}`;
    case "mcp":
      return `mcp ${toolCall.original.server}/${toolCall.original.tool}`;
    default:
      return toolCall.name;
  }
}

function toolCallLocations(toolCall: ToolCallRequest, cwd: string): ToolCallLocation[] {
  // Relative paths resolve against the session cwd, as LocalTransport does.
  const at = (file: string, line?: number): ToolCallLocation[] => {
    const absolute = path.isAbsolute(file) ? path.normalize(file) : path.join(cwd, file);
    return [line == null ? { path: absolute } : { path: absolute, line }];
  };
  if (toolCall.name === "partial-read") {
    return at(toolCall.original.filePath, toolCall.original.offset);
  }
  if ("filePath" in toolCall.original) return at(toolCall.original.filePath);
  switch (toolCall.name) {
    case "list":
      return at(toolCall.original.dirPath || ".");
    case "glob":
    case "grep":
      return toolCall.original.path == null ? [] : at(toolCall.original.path);
    default:
      return [];
  }
}

// rawInput is the model's original arguments: parsed ones can include whole file contents.
export function toolCallStart(toolCall: ToolCallRequest, cwd: string) {
  return {
    toolCallId: toolCall.toolCallId,
    title: toolCallTitle(toolCall),
    kind: toolKind(toolCall.name),
    status: "pending" as const,
    rawInput: toolCall.original,
    locations: toolCallLocations(toolCall, cwd),
  };
}

type ToolCallResult = {
  toolCallId: string;
  status: "completed" | "failed";
  content: ToolCallContent[];
};

export function toolCallResult(ir: OctoIR): ToolCallResult | null {
  const result = (toolCallId: string, status: ToolCallResult["status"], text: string) => ({
    toolCallId,
    status,
    content: text === "" ? [] : [textContent(text)],
  });
  switch (ir.role) {
    case "tool-output": {
      return result(ir.toolCall.toolCallId, "completed", contentText(ir.content));
    }
    case "file-read":
    case "file-mutate":
      return result(ir.toolCall.toolCallId, "completed", ir.content);
    case "tool-runtime-error":
    case "tool-validation-error":
      return result(ir.toolCall.toolCallId, "failed", ir.error);
    case "tool-skip-output":
      return result(ir.toolCall.toolCallId, "failed", ir.reason);
    case "tool-reject":
      return result(ir.toolCall.toolCallId, "failed", "Rejected by the user");
    default:
      return null;
  }
}

function textContent(text: string): ToolCallContent {
  return { type: "content", content: { type: "text", text } };
}
