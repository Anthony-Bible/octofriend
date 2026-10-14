import { fileURLToPath } from "url";
import type { ContentBlock } from "@agentclientprotocol/sdk";
import type { UserMessage } from "../libocto/llm-ir.ts";
import { err, ok, type Result } from "../libocto/result.ts";

// Only text and resource links are supported (promptCapabilities is empty). A link becomes an
// @-mention; the agent reads the file with its own tools.
export function promptToUserContent(
  blocks: readonly ContentBlock[],
): Result<UserMessage["content"], string> {
  const content: UserMessage["content"] = [];
  for (const block of blocks) {
    switch (block.type) {
      case "text":
        content.push({ type: "text", content: block.text });
        break;
      case "resource_link":
        content.push({ type: "text", content: resourceLinkText(block.name, block.uri) });
        break;
      case "image":
      case "audio":
      case "resource":
        return err(`Unsupported prompt content type: ${block.type}`);
    }
  }
  return ok(content);
}

function resourceLinkText(name: string, uri: string): string {
  const filePath = uri.startsWith("file://") ? localPath(uri) : uri;
  return name === "" || filePath.endsWith(name) ? `@${filePath}` : `@${filePath} (${name})`;
}

// Falls back to the raw URI when it isn't a local path (remote host, bad escape).
function localPath(uri: string): string {
  try {
    return fileURLToPath(uri);
  } catch {
    return uri;
  }
}
