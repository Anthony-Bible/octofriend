import fs from "fs/promises";
import path from "path";
import { spawn, execFile, type ChildProcess } from "child_process";
import {
  Transport,
  TransportError,
  runShell,
  runBackgroundShell,
  type BackgroundShellOptions,
} from "./transport-common.ts";
import { processes } from "../process-manager.ts";
import {
  TransportProcess,
  type TransportProcessOptions,
  type ProcessSpawnOptions,
  type ProcessExecFileOptions,
  type ProcessExecFileCallback,
} from "./transport-process.ts";

const KILL_GRACE_MS = 500;

const STRIPPED_ENV_VARS = ["NODE_ENV", "NAPI_RS_NATIVE_LIBRARY_PATH", "CANARY_OCTO"];

export class LocalTransport implements Transport {
  readonly cwd: string;
  // bash over sh: available on most local setups, and tolerant of LLM bash-isms
  readonly commandShell = "bash";
  private readonly runningProcesses = new Set<TransportProcess>();
  private readonly processManager = processes.manager();

  // Relative paths resolve against cwd, not process.cwd(), so a transport can serve another
  // project.
  constructor(cwd = process.cwd()) {
    this.cwd = cwd;
  }

  private abs(file: string): string {
    return path.resolve(this.cwd, file);
  }

  async close() {
    await Promise.all(
      [...this.runningProcesses].map(localProcess =>
        localProcess.terminate({ graceMs: KILL_GRACE_MS }),
      ),
    );
  }

  spawn(command: string, args: readonly string[], options: ProcessSpawnOptions): TransportProcess {
    const { surviveAfterOctoExit, ...spawnOptions } = options;
    return this.manage(
      spawn(command, args, {
        cwd: this.cwd,
        env: commandEnvironment(),
        ...spawnOptions,
      }),
      { detached: options.detached, surviveAfterOctoExit },
    );
  }

  execFile(
    file: string,
    args: readonly string[],
    options: ProcessExecFileOptions,
    callback: ProcessExecFileCallback | undefined,
  ): TransportProcess {
    const localProcess = this.manage(
      execFile(
        file,
        args,
        {
          cwd: this.cwd,
          env: commandEnvironment(),
          ...options,
          encoding: "utf8",
        },
        callback,
      ),
      {},
    );
    localProcess.on("error", () => {});
    return localProcess;
  }

  private manage(childProcess: ChildProcess, options: TransportProcessOptions): TransportProcess {
    const localProcess = new TransportProcess(childProcess, options);
    this.runningProcesses.add(localProcess);
    this.processManager.register(localProcess);
    void localProcess.processClosedPromise.then(() => this.runningProcesses.delete(localProcess));
    return localProcess;
  }

  async writeFile(_: AbortSignal, file: string, contents: string) {
    return await fs.writeFile(this.abs(file), contents, "utf8");
  }

  async readFile(_: AbortSignal, file: string) {
    return await fs.readFile(this.abs(file), "utf8");
  }

  async modTime(_: AbortSignal, file: string) {
    try {
      const stat = await fs.stat(this.abs(file));
      return stat.mtimeMs;
    } catch (e) {
      throw new TransportError(`Could not get modified time for ${file}: ${e}`);
    }
  }

  async resolvePath(_: AbortSignal, file: string) {
    try {
      return await fs.realpath(this.abs(file));
    } catch {
      return this.abs(file);
    }
  }

  async mkdir(_: AbortSignal, dirpath: string) {
    await fs.mkdir(this.abs(dirpath), { recursive: true });
  }

  async readdir(_: AbortSignal, dirpath: string) {
    const dir = this.abs(dirpath);
    const entries = await fs.readdir(dir, { withFileTypes: true });
    return Promise.all(
      entries.map(async entry => {
        // For symlinks, resolve to determine if target is a directory
        if (entry.isSymbolicLink()) {
          const fullPath = path.join(dir, entry.name);
          try {
            const stat = await fs.stat(fullPath); // follows symlinks
            return { entry: entry.name, isDirectory: stat.isDirectory() };
          } catch {
            // Broken symlink or permission error - treat as file
            return { entry: entry.name, isDirectory: false };
          }
        }
        return { entry: entry.name, isDirectory: entry.isDirectory() };
      }),
    );
  }

  async pathExists(signal: AbortSignal, file: string) {
    try {
      await this.modTime(signal, file);
      return true;
    } catch {
      return false;
    }
  }

  async isDirectory(_: AbortSignal, file: string) {
    try {
      const stat = await fs.stat(this.abs(file));
      return stat.isDirectory();
    } catch {
      return false;
    }
  }

  backgroundShell(options: BackgroundShellOptions) {
    return runBackgroundShell(this, options);
  }

  async shell(signal: AbortSignal, cmd: string, timeout: number) {
    return runShell(this, signal, cmd, timeout);
  }
}

function commandEnvironment(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  for (const name of STRIPPED_ENV_VARS) delete env[name];
  return env;
}
