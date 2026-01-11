/**
 * ComparisonAdapter - Evaluates LLM performance with and without mochi context.
 *
 * Runs two modes for each query:
 * 1. Baseline: LLM generates code without domain context
 * 2. Assisted: LLM generates code with mochi-provided context
 *
 * Compares results to measure mochi's effectiveness in reducing hallucinations.
 */

import type {
  SearchAdapter,
  SearchAdapterContext,
} from "../../../vendor/assay-kit/packages/assay-kit/src/types/adapters.js";
import type { Query } from "../../../vendor/assay-kit/packages/assay-kit/src/types/query.js";
import type { Dataset } from "../../../vendor/assay-kit/packages/assay-kit/src/types/dataset.js";
import { OllamaClient, type OllamaGenerateResult } from "./ollama-client.js";
import { MCPClient } from "./mcp-client.js";

/**
 * Query format for comparison evaluation
 */
export interface ComparisonQuery extends Query {
  payload: {
    instruction: string;
    input: string;
    context?: string;
    expectedKeywords?: string[];
    forbiddenKeywords?: string[];
    expectedPattern?: string;
  };
}

/**
 * Single evaluation result
 */
export interface EvaluationResult {
  response: string;
  latencyMs: number;
  tokensGenerated: number;
  keywordScore: number;
  hallucinationScore: number;
  patternMatched: boolean | null;
  foundKeywords: string[];
  missingKeywords: string[];
  violations: string[];
}

/**
 * Comparison metrics for a single query
 */
export interface ComparisonMetrics {
  baseline: EvaluationResult;
  assisted: EvaluationResult;
  mochiOnly?: EvaluationResult;
  improvement: {
    qualityDelta: number;
    hallucinationDelta: number;
    latencyDelta: number;
  };
}

export interface ComparisonAdapterOptions {
  /** Ollama endpoint (default: http://localhost:11434) */
  ollamaEndpoint?: string;
  /** Ollama model name (default: gpt-oss:120b) */
  ollamaModel?: string;
  /** Python command for mochi MCP server */
  pythonCommand?: string;
  /** Whether to also run mochi-only evaluation */
  includeMochiOnly?: boolean;
}

export class ComparisonAdapter
  implements SearchAdapter<ComparisonQuery, ComparisonMetrics>
{
  private ollamaClient: OllamaClient;
  private mcpClient: MCPClient;
  private ollamaModel: string;
  private includeMochiOnly: boolean;

  constructor(options: ComparisonAdapterOptions = {}) {
    this.ollamaClient = new OllamaClient({
      endpoint: options.ollamaEndpoint ?? "http://localhost:11434",
      defaultModel: options.ollamaModel ?? "gpt-oss:120b",
    });
    this.mcpClient = new MCPClient(options.pythonCommand ?? "python3");
    this.ollamaModel = options.ollamaModel ?? "gpt-oss:120b";
    this.includeMochiOnly = options.includeMochiOnly ?? false;
  }

  /**
   * Warmup: verify both Ollama and MCP connections
   */
  async warmup(_dataset: Dataset<ComparisonQuery>): Promise<void> {
    console.log("[ComparisonAdapter] Warming up...");

    // Check Ollama
    console.log("[ComparisonAdapter] Checking Ollama connection...");
    const ollamaOk = await this.ollamaClient.ping();
    if (!ollamaOk) {
      throw new Error("Ollama server not available. Run: ollama serve");
    }

    // Check if model is available
    const models = await this.ollamaClient.listModels();
    if (!models.some(m => m.includes(this.ollamaModel.split(":")[0]))) {
      throw new Error(
        `Model ${this.ollamaModel} not found. Available: ${models.join(", ")}`
      );
    }
    console.log(`[ComparisonAdapter] Ollama OK, using model: ${this.ollamaModel}`);

    // Check MCP
    console.log("[ComparisonAdapter] Checking MCP connection...");
    const mcpOk = await this.mcpClient.ping();
    if (!mcpOk) {
      throw new Error("Failed to connect to mochi MCP server");
    }
    console.log("[ComparisonAdapter] MCP connection established");

    // Warmup LLM with a simple prompt
    console.log("[ComparisonAdapter] Warming up LLM...");
    await this.ollamaClient.generate("Hello", {
      model: this.ollamaModel,
      maxTokens: 10,
    });
    console.log("[ComparisonAdapter] Warmup complete");
  }

  /**
   * Execute comparison evaluation for a single query
   */
  async execute(
    query: ComparisonQuery,
    ctx: SearchAdapterContext
  ): Promise<ComparisonMetrics> {
    if (ctx.signal.aborted) {
      throw new Error(`Query ${query.id} was cancelled`);
    }

    // 1. Baseline: LLM without context
    console.log(`[${query.id}] Running baseline (LLM only)...`);
    const baselineResult = await this.runLLM(query, null);
    const baseline = this.evaluate(baselineResult, query);

    if (ctx.signal.aborted) {
      throw new Error(`Query ${query.id} was cancelled`);
    }

    // 2. Assisted: LLM with mochi SLM response
    console.log(`[${query.id}] Running assisted (LLM + mochi SLM)...`);

    // Call mochi's SLM to get context
    const mochiContext = await this.getMochiContext(query);
    console.log(`[${query.id}] Mochi SLM response: ${mochiContext.substring(0, 100)}...`);

    const assistedResult = await this.runLLM(query, mochiContext);
    const assisted = this.evaluate(assistedResult, query);

    // 3. Optional: mochi-only (SLM direct output without LLM)
    let mochiOnly: EvaluationResult | undefined;
    if (this.includeMochiOnly) {
      console.log(`[${query.id}] Running mochi-only...`);
      const mochiResult = await this.runMochiOnly(query);
      mochiOnly = this.evaluate(mochiResult, query);
    }

    // Calculate improvement
    const improvement = {
      qualityDelta: assisted.keywordScore - baseline.keywordScore,
      hallucinationDelta: baseline.hallucinationScore - assisted.hallucinationScore,
      latencyDelta: assisted.latencyMs - baseline.latencyMs,
    };

    return {
      baseline,
      assisted,
      mochiOnly,
      improvement,
    };
  }

  /**
   * Get context from mochi's SLM via MCP
   */
  private async getMochiContext(query: ComparisonQuery): Promise<string> {
    try {
      const result = await this.mcpClient.call("domain_query", {
        instruction: query.payload.instruction,
        input: query.payload.input,
        context: "", // No pre-existing context - let SLM generate from its training
        validate: false,
        mode: "auto",
      });

      // Return the SLM's response as context for the LLM
      return result.response;
    } catch (error) {
      console.error(`[${query.id}] Mochi SLM call failed:`, error);
      // Fallback to empty context if MCP fails
      return "";
    }
  }

  /**
   * Run LLM with optional context
   */
  private async runLLM(
    query: ComparisonQuery,
    context: string | null
  ): Promise<{ response: string; latencyMs: number; tokensGenerated: number }> {
    const startTime = Date.now();

    let prompt = `## Task\n${query.payload.instruction}\n\n## Code\n${query.payload.input}`;

    if (context) {
      prompt = `## Domain Context (use this information)\n${context}\n\n${prompt}`;
    }

    prompt += "\n\n## Your completion (code only):";

    const result = await this.ollamaClient.generate(prompt, {
      model: this.ollamaModel,
      temperature: 0.1,
      maxTokens: 256,
    });

    return {
      response: result.response.trim(),
      latencyMs: Date.now() - startTime,
      tokensGenerated: result.evalCount,
    };
  }

  /**
   * Run mochi MCP server directly
   */
  private async runMochiOnly(
    query: ComparisonQuery
  ): Promise<{ response: string; latencyMs: number; tokensGenerated: number }> {
    const startTime = Date.now();

    const result = await this.mcpClient.call("domain_query", {
      instruction: query.payload.instruction,
      input: query.payload.input,
      context: query.payload.context ?? "",
      validate: true,
      mode: "auto",
    });

    return {
      response: result.response,
      latencyMs: Date.now() - startTime,
      tokensGenerated: result.tokens_generated,
    };
  }

  /**
   * Evaluate a single result against query expectations
   */
  private evaluate(
    result: { response: string; latencyMs: number; tokensGenerated: number },
    query: ComparisonQuery
  ): EvaluationResult {
    const expected = query.payload.expectedKeywords ?? [];
    const forbidden = query.payload.forbiddenKeywords ?? [];
    const output = result.response.toLowerCase();

    // Check expected keywords
    const foundKeywords = expected.filter(k =>
      output.includes(k.toLowerCase())
    );
    const missingKeywords = expected.filter(
      k => !output.includes(k.toLowerCase())
    );

    // Check forbidden keywords (hallucinations)
    const violations = forbidden.filter(k =>
      output.includes(k.toLowerCase())
    );

    // Check pattern
    let patternMatched: boolean | null = null;
    if (query.payload.expectedPattern) {
      try {
        patternMatched = new RegExp(query.payload.expectedPattern).test(
          result.response
        );
      } catch {
        patternMatched = false;
      }
    }

    // Calculate scores
    let keywordScore = 1.0;
    if (expected.length > 0) {
      keywordScore = foundKeywords.length / expected.length;
    }
    if (patternMatched === false) {
      keywordScore *= 0.8;
    }

    const hallucinationScore = Math.min(1, violations.length * 0.25);

    return {
      response: result.response,
      latencyMs: result.latencyMs,
      tokensGenerated: result.tokensGenerated,
      keywordScore: Math.max(0, Math.min(1, keywordScore)),
      hallucinationScore,
      patternMatched,
      foundKeywords,
      missingKeywords,
      violations,
    };
  }

  /**
   * Cleanup: disconnect from servers
   */
  async stop(): Promise<void> {
    console.log("[ComparisonAdapter] Stopping...");
    await this.mcpClient.disconnect();
  }
}

/**
 * Summary statistics for comparison report
 */
export interface ComparisonSummary {
  timestamp: string;
  model: string;
  totalQueries: number;
  summary: {
    baselineAvgQuality: number;
    assistedAvgQuality: number;
    improvementPercent: number;
    baselineAvgHallucination: number;
    assistedAvgHallucination: number;
    hallucinationReductionPercent: number;
  };
  byCategory: Record<
    string,
    {
      count: number;
      baselineAvgQuality: number;
      assistedAvgQuality: number;
      improvement: number;
    }
  >;
}

/**
 * Generate comparison summary from results
 */
export function generateComparisonSummary(
  results: Array<{ queryId: string; metrics: ComparisonMetrics; category?: string }>,
  model: string
): ComparisonSummary {
  const totalQueries = results.length;

  let baselineTotalQuality = 0;
  let assistedTotalQuality = 0;
  let baselineTotalHallucination = 0;
  let assistedTotalHallucination = 0;

  const byCategory: ComparisonSummary["byCategory"] = {};

  for (const r of results) {
    baselineTotalQuality += r.metrics.baseline.keywordScore;
    assistedTotalQuality += r.metrics.assisted.keywordScore;
    baselineTotalHallucination += r.metrics.baseline.hallucinationScore;
    assistedTotalHallucination += r.metrics.assisted.hallucinationScore;

    const cat = r.category ?? "unknown";
    if (!byCategory[cat]) {
      byCategory[cat] = {
        count: 0,
        baselineAvgQuality: 0,
        assistedAvgQuality: 0,
        improvement: 0,
      };
    }
    byCategory[cat].count++;
    byCategory[cat].baselineAvgQuality += r.metrics.baseline.keywordScore;
    byCategory[cat].assistedAvgQuality += r.metrics.assisted.keywordScore;
  }

  // Calculate averages
  const baselineAvgQuality = baselineTotalQuality / totalQueries;
  const assistedAvgQuality = assistedTotalQuality / totalQueries;
  const baselineAvgHallucination = baselineTotalHallucination / totalQueries;
  const assistedAvgHallucination = assistedTotalHallucination / totalQueries;

  const improvementPercent =
    baselineAvgQuality > 0
      ? ((assistedAvgQuality - baselineAvgQuality) / baselineAvgQuality) * 100
      : 0;

  const hallucinationReductionPercent =
    baselineAvgHallucination > 0
      ? ((baselineAvgHallucination - assistedAvgHallucination) /
          baselineAvgHallucination) *
        100
      : 0;

  // Calculate category averages
  for (const cat of Object.keys(byCategory)) {
    const data = byCategory[cat];
    data.baselineAvgQuality /= data.count;
    data.assistedAvgQuality /= data.count;
    data.improvement =
      data.baselineAvgQuality > 0
        ? ((data.assistedAvgQuality - data.baselineAvgQuality) /
            data.baselineAvgQuality) *
          100
        : 0;
  }

  return {
    timestamp: new Date().toISOString(),
    model,
    totalQueries,
    summary: {
      baselineAvgQuality,
      assistedAvgQuality,
      improvementPercent,
      baselineAvgHallucination,
      assistedAvgHallucination,
      hallucinationReductionPercent,
    },
    byCategory,
  };
}
