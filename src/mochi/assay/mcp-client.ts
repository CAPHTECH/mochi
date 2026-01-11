/**
 * MCP Client for stdio communication with mochi server.
 *
 * Spawns python mochi.mcp.server and communicates via JSON-RPC over stdio.
 */

import { spawn, type ChildProcess } from "child_process";
import { createInterface, type Interface } from "readline";

interface MCPRequest {
  jsonrpc: "2.0";
  id: number;
  method: string;
  params?: Record<string, unknown>;
}

interface MCPResponse {
  jsonrpc: "2.0";
  id: number;
  result?: unknown;
  error?: {
    code: number;
    message: string;
    data?: unknown;
  };
}

interface DomainQueryResult {
  response: string;
  confidence: number;
  inference_time_ms: number;
  tokens_generated: number;
  mode_used: string;
  retried: boolean;
  warning?: string;
  validation?: {
    is_valid: boolean;
    hallucination_rate: number;
    hallucinated_methods: string[];
    suggestions: Record<string, string>;
  };
}

export class MCPClient {
  private process: ChildProcess | null = null;
  private readline: Interface | null = null;
  private requestId = 0;
  private pendingRequests = new Map<
    number,
    {
      resolve: (value: unknown) => void;
      reject: (error: Error) => void;
    }
  >();
  private connected = false;

  constructor(private pythonCommand: string = "python3") {}

  /**
   * Start the MCP server process
   */
  private connecting: Promise<void> | null = null;

  async connect(): Promise<void> {
    if (this.connected) return;
    if (this.connecting) return this.connecting;

    this.connecting = this.doConnect();
    try {
      await this.connecting;
    } finally {
      this.connecting = null;
    }
  }

  private async doConnect(): Promise<void> {
    // Use JSON.stringify to escape the path safely (prevents command injection)
    const srcPath = JSON.stringify(`${process.cwd()}/src`);
    const pythonCode = `
import sys
sys.path.insert(0, ${srcPath})
from mochi.mcp.server import MochiMCPServer
server = MochiMCPServer()
try:
    server.run_stdio()
finally:
    server.shutdown()
`.trim();

    this.process = spawn(this.pythonCommand, ["-c", pythonCode], {
      cwd: process.cwd(),
      env: process.env,
      stdio: ["pipe", "pipe", "pipe"],
    });

    if (!this.process.stdout || !this.process.stdin) {
      throw new Error("Failed to create stdio streams");
    }

    this.readline = createInterface({
      input: this.process.stdout,
      crlfDelay: Infinity,
    });

    this.readline.on("line", (line) => {
      this.handleResponse(line);
    });

    this.process.stderr?.on("data", (data) => {
      // Log stderr for debugging but don't fail
      console.error(`[MCP stderr] ${data.toString()}`);
    });

    this.process.on("error", (error) => {
      console.error(`[MCP process error] ${error.message}`);
      this.connected = false;
    });

    this.process.on("exit", (code) => {
      console.log(`[MCP process exited] code=${code}`);
      this.connected = false;
    });

    // Wait for server to be ready
    await this.waitForReady();
    this.connected = true;
  }

  /**
   * Wait for MCP server to be ready
   */
  private async waitForReady(): Promise<void> {
    // Give the server a moment to start
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // Initialize the connection
    await this.sendRequest("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: {
        name: "mochi-assay-adapter",
        version: "1.0.0",
      },
    });

    // Send initialized notification
    this.sendNotification("notifications/initialized", {});
  }

  /**
   * Handle incoming response line
   */
  private handleResponse(line: string): void {
    if (!line.trim()) return;

    try {
      const response = JSON.parse(line) as MCPResponse;

      if (response.id !== undefined) {
        const pending = this.pendingRequests.get(response.id);
        if (pending) {
          this.pendingRequests.delete(response.id);

          if (response.error) {
            pending.reject(
              new Error(`MCP error: ${response.error.message}`)
            );
          } else {
            pending.resolve(response.result);
          }
        }
      }
    } catch (e) {
      // Not valid JSON, might be log output
      console.log(`[MCP output] ${line}`);
    }
  }

  /**
   * Send a request and wait for response
   */
  private async sendRequest(
    method: string,
    params?: Record<string, unknown>
  ): Promise<unknown> {
    if (!this.process?.stdin) {
      throw new Error("MCP client not connected");
    }

    const id = ++this.requestId;
    const request: MCPRequest = {
      jsonrpc: "2.0",
      id,
      method,
      params,
    };

    return new Promise((resolve, reject) => {
      // Timeout after 120 seconds (inference can be slow)
      const timeoutId = setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id);
          reject(new Error(`Request ${method} timed out`));
        }
      }, 120000);

      this.pendingRequests.set(id, {
        resolve: (value) => {
          clearTimeout(timeoutId);
          resolve(value);
        },
        reject: (error) => {
          clearTimeout(timeoutId);
          reject(error);
        },
      });

      const json = JSON.stringify(request);
      this.process!.stdin!.write(json + "\n");
    });
  }

  /**
   * Send a notification (no response expected)
   */
  private sendNotification(
    method: string,
    params?: Record<string, unknown>
  ): void {
    if (!this.process?.stdin) return;

    const notification = {
      jsonrpc: "2.0",
      method,
      params,
    };

    this.process.stdin.write(JSON.stringify(notification) + "\n");
  }

  /**
   * Check if server is alive
   */
  async ping(): Promise<boolean> {
    try {
      await this.connect();
      // List tools to verify connection
      await this.sendRequest("tools/list", {});
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Call a tool on the MCP server
   */
  async call(
    toolName: string,
    args: Record<string, unknown>
  ): Promise<DomainQueryResult> {
    await this.connect();

    const result = await this.sendRequest("tools/call", {
      name: toolName,
      arguments: args,
    });

    // Runtime validation of response structure
    if (!result || typeof result !== "object") {
      throw new Error("Invalid response: expected object");
    }
    const obj = result as Record<string, unknown>;
    if (!Array.isArray(obj.content)) {
      throw new Error("Invalid response: missing content array");
    }

    // Find text content with proper type checking
    const textContent = obj.content.find(
      (c: unknown): c is { type: string; text: string } =>
        typeof c === "object" &&
        c !== null &&
        (c as Record<string, unknown>).type === "text" &&
        typeof (c as Record<string, unknown>).text === "string"
    );

    if (!textContent) {
      throw new Error("No text content in response");
    }

    return JSON.parse(textContent.text) as DomainQueryResult;
  }

  /**
   * Disconnect from the MCP server
   */
  async disconnect(): Promise<void> {
    if (this.readline) {
      this.readline.close();
      this.readline = null;
    }

    if (this.process) {
      this.process.kill();
      this.process = null;
    }

    this.connected = false;
    this.pendingRequests.clear();
  }
}
