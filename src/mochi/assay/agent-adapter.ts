/**
 * AgentAdapter - Evaluates LLM as an agent with tool-calling capabilities.
 *
 * Simulates a realistic agent workflow:
 * 1. Agent receives task (no context provided upfront)
 * 2. Agent decides whether to call mochi tools
 * 3. If tool called, agent receives context
 * 4. Agent generates code using retrieved context
 *
 * Evaluates:
 * - Tool call decision accuracy (should it have called mochi?)
 * - Context utilization (did it use the retrieved info?)
 * - Final code quality
 */

import type {
  SearchAdapter,
  SearchAdapterContext,
} from "../../../vendor/assay-kit/packages/assay-kit/src/types/adapters.js";
import type { Query } from "../../../vendor/assay-kit/packages/assay-kit/src/types/query.js";
import type { Dataset } from "../../../vendor/assay-kit/packages/assay-kit/src/types/dataset.js";
import { OllamaClient } from "./ollama-client.js";
import { MCPClient } from "./mcp-client.js";

/**
 * Query format for agent evaluation
 */
export interface AgentQuery extends Query {
  payload: {
    instruction: string;
    input: string;
    /** Ground truth context (for evaluation, not provided to agent) */
    context?: string;
    expectedKeywords?: string[];
    forbiddenKeywords?: string[];
    expectedPattern?: string;
  };
  metadata?: {
    category?: string;
    /** Should the agent call mochi tools for this query? */
    shouldCallTool?: boolean;
    /** Expected tool to call */
    expectedTool?: string;
    difficulty?: string;
  };
}

/**
 * Tool call parsed from agent response
 */
export interface ParsedToolCall {
  toolName: string;
  arguments: Record<string, unknown>;
}

/**
 * Agent step in the execution trace
 */
export interface AgentStep {
  type: "thinking" | "tool_call" | "tool_result" | "generation";
  content: string;
  timestamp: number;
}

/**
 * Metrics for agent evaluation
 */
export interface AgentMetrics {
  /** Final generated response */
  response: string;
  /** Total execution time */
  latencyMs: number;
  /** Tokens generated across all steps */
  tokensGenerated: number;

  /** Tool calling evaluation */
  toolDecision: {
    /** Did the agent decide to call a tool? */
    calledTool: boolean;
    /** Was the decision correct? */
    decisionCorrect: boolean;
    /** Which tool was called (if any) */
    toolCalled?: string;
    /** Tool call arguments */
    toolArgs?: Record<string, unknown>;
  };

  /** Context utilization */
  contextUtilization: {
    /** Did the agent receive context from tool? */
    receivedContext: boolean;
    /** Did the output use info from context? */
    usedContext: boolean;
    /** Score of how well context was used (0-1) */
    utilizationScore: number;
  };

  /** Code quality (same as comparison adapter) */
  codeQuality: {
    keywordScore: number;
    hallucinationScore: number;
    patternMatched: boolean | null;
    foundKeywords: string[];
    missingKeywords: string[];
    violations: string[];
  };

  /** Full execution trace */
  trace: AgentStep[];
}

export interface AgentAdapterOptions {
  ollamaEndpoint?: string;
  ollamaModel?: string;
  pythonCommand?: string;
  /** Max iterations for agent loop */
  maxIterations?: number;
}

/**
 * Tool definitions for the agent
 */
const MOCHI_TOOLS = `
## Available Tools

### mochi_domain_query
Get domain-specific information about APIs, patterns, and conventions.
Use this when you need to know:
- What methods are available on a class/object
- Project-specific coding patterns
- Type definitions and interfaces

Arguments:
- query: string - Description of what information you need

Example:
<tool_call>
mochi_domain_query(query="What methods are available on KiriMCPClient?")
</tool_call>

### mochi_suggest_pattern
Get suggested code patterns for a specific goal.

Arguments:
- goal: string - What you're trying to achieve
- context: string (optional) - Current code context

Example:
<tool_call>
mochi_suggest_pattern(goal="error handling for MCP tool calls")
</tool_call>

## Instructions
- If you're unsure about APIs or project conventions, USE THE TOOLS
- If you know the answer confidently, you can skip tool calls
- After tool results, generate your final code
`;

export class AgentAdapter implements SearchAdapter<AgentQuery, AgentMetrics> {
  private ollamaClient: OllamaClient;
  private mcpClient: MCPClient;
  private ollamaModel: string;
  private maxIterations: number;

  constructor(options: AgentAdapterOptions = {}) {
    this.ollamaClient = new OllamaClient({
      endpoint: options.ollamaEndpoint ?? "http://localhost:11434",
      defaultModel: options.ollamaModel ?? "gpt-oss:120b",
    });
    this.mcpClient = new MCPClient(options.pythonCommand ?? "python3");
    this.ollamaModel = options.ollamaModel ?? "gpt-oss:120b";
    this.maxIterations = options.maxIterations ?? 3;
  }

  async warmup(_dataset: Dataset<AgentQuery>): Promise<void> {
    console.log("[AgentAdapter] Warming up...");

    // Check Ollama
    const ollamaOk = await this.ollamaClient.ping();
    if (!ollamaOk) {
      throw new Error("Ollama server not available");
    }
    console.log("[AgentAdapter] Ollama OK");

    // Check MCP
    const mcpOk = await this.mcpClient.ping();
    if (!mcpOk) {
      throw new Error("MCP server not available");
    }
    console.log("[AgentAdapter] MCP OK");

    // Warmup LLM
    await this.ollamaClient.generate("Hello", {
      model: this.ollamaModel,
      maxTokens: 10,
    });
    console.log("[AgentAdapter] Warmup complete");
  }

  async execute(
    query: AgentQuery,
    ctx: SearchAdapterContext
  ): Promise<AgentMetrics> {
    const startTime = Date.now();
    const trace: AgentStep[] = [];
    let totalTokens = 0;

    // Build initial prompt (NO context provided)
    const systemPrompt = `You are a code completion assistant with access to domain-specific tools.

${MOCHI_TOOLS}

## Response Format
1. First, decide if you need to call a tool
2. If yes, output: <tool_call>tool_name(args)</tool_call>
3. After receiving tool results (or if no tool needed), output your code
4. Mark final code with: <code>your code here</code>
`;

    const userPrompt = `## Task
${query.payload.instruction}

## Code to Complete
${query.payload.input}

Think step by step. Do you need domain information? If yes, call a tool first.`;

    let conversationHistory = [
      { role: "system" as const, content: systemPrompt },
      { role: "user" as const, content: userPrompt },
    ];

    let toolCalled = false;
    let toolName: string | undefined;
    let toolArgs: Record<string, unknown> | undefined;
    let receivedContext = false;
    let mochiContext = "";
    let finalResponse = "";

    // Agent loop
    for (let i = 0; i < this.maxIterations; i++) {
      if (ctx.signal.aborted) {
        throw new Error(`Query ${query.id} was cancelled`);
      }

      const result = await this.ollamaClient.chat(conversationHistory, {
        model: this.ollamaModel,
        temperature: 0.1,
        maxTokens: 512,
      });

      totalTokens += result.evalCount;
      const response = result.response;

      trace.push({
        type: "thinking",
        content: response,
        timestamp: Date.now() - startTime,
      });

      // Check for tool call
      const toolCall = this.parseToolCall(response);
      if (toolCall && !toolCalled) {
        toolCalled = true;
        toolName = toolCall.toolName;
        toolArgs = toolCall.arguments;

        trace.push({
          type: "tool_call",
          content: `${toolCall.toolName}(${JSON.stringify(toolCall.arguments)})`,
          timestamp: Date.now() - startTime,
        });

        // Execute tool via MCP
        try {
          const toolResult = await this.executeToolCall(toolCall);
          mochiContext = toolResult;
          receivedContext = true;

          trace.push({
            type: "tool_result",
            content: toolResult,
            timestamp: Date.now() - startTime,
          });

          // Add to conversation
          conversationHistory.push({
            role: "assistant" as const,
            content: response,
          });
          conversationHistory.push({
            role: "user" as const,
            content: `<tool_result>\n${toolResult}\n</tool_result>\n\nNow generate the code based on this information.`,
          });

          continue; // Next iteration
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : String(error);
          trace.push({
            type: "tool_result",
            content: `Error: ${errorMsg}`,
            timestamp: Date.now() - startTime,
          });
        }
      }

      // Check for final code
      const codeMatch = response.match(/<code>([\s\S]*?)<\/code>/);
      if (codeMatch) {
        finalResponse = codeMatch[1].trim();
        trace.push({
          type: "generation",
          content: finalResponse,
          timestamp: Date.now() - startTime,
        });
        break;
      }

      // If no tool call and no code block, treat the response as code
      if (!toolCall) {
        // Try to extract code-like content
        finalResponse = this.extractCode(response);
        trace.push({
          type: "generation",
          content: finalResponse,
          timestamp: Date.now() - startTime,
        });
        break;
      }
    }

    const latencyMs = Date.now() - startTime;

    // Evaluate tool decision
    const shouldCallTool = query.metadata?.shouldCallTool ?? true; // Default: should call
    const decisionCorrect = toolCalled === shouldCallTool;

    // Evaluate context utilization
    let usedContext = false;
    let utilizationScore = 0;
    if (receivedContext && mochiContext) {
      // Check if keywords from context appear in output
      const contextKeywords = this.extractKeywords(mochiContext);
      const usedKeywords = contextKeywords.filter((k) =>
        finalResponse.toLowerCase().includes(k.toLowerCase())
      );
      usedContext = usedKeywords.length > 0;
      utilizationScore =
        contextKeywords.length > 0 ? usedKeywords.length / contextKeywords.length : 0;
    }

    // Evaluate code quality
    const codeQuality = this.evaluateCode(finalResponse, query);

    return {
      response: finalResponse,
      latencyMs,
      tokensGenerated: totalTokens,
      toolDecision: {
        calledTool: toolCalled,
        decisionCorrect,
        toolCalled: toolName,
        toolArgs,
      },
      contextUtilization: {
        receivedContext,
        usedContext,
        utilizationScore,
      },
      codeQuality,
      trace,
    };
  }

  /**
   * Parse tool call from agent response
   */
  private parseToolCall(response: string): ParsedToolCall | null {
    const match = response.match(/<tool_call>\s*(\w+)\s*\((.*?)\)\s*<\/tool_call>/s);
    if (!match) return null;

    const toolName = match[1];
    const argsStr = match[2];

    // Parse arguments
    const args: Record<string, unknown> = {};
    const argMatches = argsStr.matchAll(/(\w+)\s*=\s*"([^"]*?)"/g);
    for (const m of argMatches) {
      args[m[1]] = m[2];
    }

    return { toolName, arguments: args };
  }

  /**
   * Execute tool call via MCP
   */
  private async executeToolCall(toolCall: ParsedToolCall): Promise<string> {
    if (toolCall.toolName === "mochi_domain_query") {
      const result = await this.mcpClient.call("domain_query", {
        instruction: "Provide API information",
        input: String(toolCall.arguments.query ?? ""),
        mode: "auto",
      });
      return result.response;
    } else if (toolCall.toolName === "mochi_suggest_pattern") {
      const result = await this.mcpClient.call("domain_query", {
        instruction: `Suggest pattern for: ${toolCall.arguments.goal}`,
        input: String(toolCall.arguments.context ?? ""),
        mode: "auto",
      });
      return result.response;
    }

    throw new Error(`Unknown tool: ${toolCall.toolName}`);
  }

  /**
   * Extract code from response without code block
   */
  private extractCode(response: string): string {
    // Remove thinking/explanation parts
    const lines = response.split("\n");
    const codeLines: string[] = [];
    let inCode = false;

    for (const line of lines) {
      // Skip obvious non-code lines
      if (line.match(/^(I |Let me|First|To |This |The |We )/i)) {
        continue;
      }
      // Start collecting when we see code-like content
      if (line.match(/^[\s]*(const |let |var |function |async |await |return |if |for |class |\w+\.\w+|\w+\s*=)/)) {
        inCode = true;
      }
      if (inCode) {
        codeLines.push(line);
      }
    }

    return codeLines.join("\n").trim() || response.trim();
  }

  /**
   * Extract keywords from context for utilization check
   */
  private extractKeywords(context: string): string[] {
    const keywords: string[] = [];

    // Extract method names
    const methodMatches = context.matchAll(/(\w+)\s*\(/g);
    for (const m of methodMatches) {
      if (m[1].length > 2) keywords.push(m[1]);
    }

    // Extract type/class names (PascalCase)
    const typeMatches = context.matchAll(/\b([A-Z][a-zA-Z]+)\b/g);
    for (const m of typeMatches) {
      keywords.push(m[1]);
    }

    return [...new Set(keywords)];
  }

  /**
   * Evaluate code quality
   */
  private evaluateCode(
    output: string,
    query: AgentQuery
  ): AgentMetrics["codeQuality"] {
    const expected = query.payload.expectedKeywords ?? [];
    const forbidden = query.payload.forbiddenKeywords ?? [];
    const outputLower = output.toLowerCase();

    const foundKeywords = expected.filter((k) =>
      outputLower.includes(k.toLowerCase())
    );
    const missingKeywords = expected.filter(
      (k) => !outputLower.includes(k.toLowerCase())
    );
    const violations = forbidden.filter((k) =>
      outputLower.includes(k.toLowerCase())
    );

    let patternMatched: boolean | null = null;
    if (query.payload.expectedPattern) {
      try {
        patternMatched = new RegExp(query.payload.expectedPattern).test(output);
      } catch {
        patternMatched = false;
      }
    }

    let keywordScore = expected.length > 0 ? foundKeywords.length / expected.length : 1.0;
    if (patternMatched === false) keywordScore *= 0.8;

    const hallucinationScore = Math.min(1, violations.length * 0.25);

    return {
      keywordScore: Math.max(0, Math.min(1, keywordScore)),
      hallucinationScore,
      patternMatched,
      foundKeywords,
      missingKeywords,
      violations,
    };
  }

  async stop(): Promise<void> {
    console.log("[AgentAdapter] Stopping...");
    await this.mcpClient.disconnect();
  }
}

/**
 * Generate summary statistics for agent evaluation
 */
export interface AgentSummary {
  timestamp: string;
  model: string;
  totalQueries: number;
  toolDecision: {
    accuracy: number;
    callRate: number;
    correctCalls: number;
    incorrectCalls: number;
  };
  contextUtilization: {
    avgUtilizationScore: number;
    usedContextRate: number;
  };
  codeQuality: {
    avgKeywordScore: number;
    avgHallucinationScore: number;
  };
  byCategory: Record<
    string,
    {
      count: number;
      toolAccuracy: number;
      avgQuality: number;
    }
  >;
}

export function generateAgentSummary(
  results: Array<{ queryId: string; category: string; metrics: AgentMetrics }>,
  model: string
): AgentSummary {
  const total = results.length;

  let correctDecisions = 0;
  let toolCalls = 0;
  let totalUtilization = 0;
  let usedContextCount = 0;
  let totalKeywordScore = 0;
  let totalHallucination = 0;

  const byCategory: AgentSummary["byCategory"] = {};

  for (const r of results) {
    if (r.metrics.toolDecision.decisionCorrect) correctDecisions++;
    if (r.metrics.toolDecision.calledTool) toolCalls++;
    totalUtilization += r.metrics.contextUtilization.utilizationScore;
    if (r.metrics.contextUtilization.usedContext) usedContextCount++;
    totalKeywordScore += r.metrics.codeQuality.keywordScore;
    totalHallucination += r.metrics.codeQuality.hallucinationScore;

    const cat = r.category;
    if (!byCategory[cat]) {
      byCategory[cat] = { count: 0, toolAccuracy: 0, avgQuality: 0 };
    }
    byCategory[cat].count++;
    if (r.metrics.toolDecision.decisionCorrect) byCategory[cat].toolAccuracy++;
    byCategory[cat].avgQuality += r.metrics.codeQuality.keywordScore;
  }

  // Calculate category averages
  for (const cat of Object.keys(byCategory)) {
    byCategory[cat].toolAccuracy /= byCategory[cat].count;
    byCategory[cat].avgQuality /= byCategory[cat].count;
  }

  return {
    timestamp: new Date().toISOString(),
    model,
    totalQueries: total,
    toolDecision: {
      accuracy: correctDecisions / total,
      callRate: toolCalls / total,
      correctCalls: correctDecisions,
      incorrectCalls: total - correctDecisions,
    },
    contextUtilization: {
      avgUtilizationScore: totalUtilization / total,
      usedContextRate: usedContextCount / total,
    },
    codeQuality: {
      avgKeywordScore: totalKeywordScore / total,
      avgHallucinationScore: totalHallucination / total,
    },
    byCategory,
  };
}
