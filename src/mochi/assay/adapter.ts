/**
 * MochiAdapter - SearchAdapter implementation for mochi evaluation.
 *
 * Connects to mochi MCP server via stdio and executes domain_query calls.
 */

import type {
  SearchAdapter,
  SearchAdapterContext,
} from "../../../vendor/assay-kit/packages/assay-kit/src/types/adapters.js";
import type { Query } from "../../../vendor/assay-kit/packages/assay-kit/src/types/query.js";
import type { Dataset } from "../../../vendor/assay-kit/packages/assay-kit/src/types/dataset.js";
import { MCPClient } from "./mcp-client.js";

/**
 * Extended query type for mochi evaluation
 */
export interface MochiQuery extends Query {
  payload: {
    /** Instruction for the model (e.g., "Fill in the typescript code") */
    instruction: string;
    /** Code context/prefix to complete */
    input: string;
    /** LSP context (available methods, types) */
    context?: string;
    /** Expected keywords in output */
    expectedKeywords?: string[];
    /** Forbidden keywords (hallucination indicators) */
    forbiddenKeywords?: string[];
    /** Regex pattern expected in output */
    expectedPattern?: string;
  };
}

/**
 * Metrics returned by mochi evaluation
 */
export interface MochiMetrics {
  /** Query latency in milliseconds */
  latencyMs: number;
  /** Model confidence score (0.0-1.0) */
  confidence: number;
  /** Quality score based on keyword/pattern matching (0.0-1.0) */
  qualityScore: number;
  /** Hallucination risk based on forbidden keywords (0.0-1.0) */
  hallucinationRisk: number;
  /** Generated response text */
  response: string;
  /** Extended metrics */
  extras: {
    /** Number of tokens generated */
    tokensGenerated: number;
    /** Generation mode used (auto, conservative, creative) */
    modeUsed: string;
    /** Whether the request was retried */
    retried: boolean;
    /** Expected keywords found */
    foundKeywords: string[];
    /** Missing expected keywords */
    missingKeywords: string[];
    /** Forbidden keywords found (violations) */
    violations: string[];
    /** Whether pattern matched */
    patternMatched: boolean | null;
    /** Validation result from mochi */
    validation?: {
      isValid: boolean;
      hallucinationRate: number;
      hallucinatedMethods: string[];
    };
  };
}

export interface MochiAdapterOptions {
  /** Python command to use (default: python3) */
  pythonCommand?: string;
}

/**
 * SearchAdapter for evaluating mochi code completion quality.
 */
export class MochiAdapter
  implements SearchAdapter<MochiQuery, MochiMetrics>
{
  private mcpClient: MCPClient;

  constructor(options: MochiAdapterOptions = {}) {
    this.mcpClient = new MCPClient(options.pythonCommand ?? "python3");
  }

  /**
   * Warmup: establish MCP connection and preload model
   */
  async warmup(_dataset: Dataset<MochiQuery>): Promise<void> {
    console.log("[MochiAdapter] Warming up - connecting to MCP server...");
    const ok = await this.mcpClient.ping();
    if (!ok) {
      throw new Error("Failed to connect to mochi MCP server");
    }
    console.log("[MochiAdapter] MCP connection established");

    // Preload model by calling a simple query
    console.log("[MochiAdapter] Preloading model (this may take a few minutes)...");
    try {
      await this.mcpClient.call("domain_query", {
        instruction: "warmup",
        input: "// warmup",
        mode: "conservative",
      });
      console.log("[MochiAdapter] Model preloaded successfully");
    } catch (e) {
      // Only ignore "empty response" errors which are expected for warmup
      const errorMessage = e instanceof Error ? e.message : String(e);
      if (errorMessage.includes("No text content")) {
        console.log("[MochiAdapter] Model preloaded (warmup returned empty)");
      } else {
        throw new Error(`Model preload failed: ${errorMessage}`);
      }
    }
  }

  /**
   * Execute a single query
   */
  async execute(
    query: MochiQuery,
    ctx: SearchAdapterContext
  ): Promise<MochiMetrics> {
    const startTime = Date.now();

    // Check for cancellation
    if (ctx.signal.aborted) {
      throw new Error(`Query ${query.id} was cancelled`);
    }

    // Call mochi domain_query
    const result = await this.mcpClient.call("domain_query", {
      instruction: query.payload.instruction,
      input: query.payload.input,
      context: query.payload.context ?? "",
      validate: true,
      mode: "auto",
    });

    const latencyMs = Date.now() - startTime;

    // Validate output against expected keywords/patterns
    const validation = this.validateOutput(result.response, query.payload);

    return {
      latencyMs,
      confidence: result.confidence,
      qualityScore: validation.qualityScore,
      hallucinationRisk: validation.hallucinationRisk,
      response: result.response,
      extras: {
        tokensGenerated: result.tokens_generated,
        modeUsed: result.mode_used,
        retried: result.retried,
        foundKeywords: validation.foundKeywords,
        missingKeywords: validation.missingKeywords,
        violations: validation.violations,
        patternMatched: validation.patternMatched,
        validation: result.validation
          ? {
              isValid: result.validation.is_valid,
              hallucinationRate: result.validation.hallucination_rate,
              hallucinatedMethods: result.validation.hallucinated_methods,
            }
          : undefined,
      },
    };
  }

  /**
   * Cleanup: disconnect from MCP server
   */
  async stop(): Promise<void> {
    console.log("[MochiAdapter] Stopping - disconnecting from MCP server...");
    await this.mcpClient.disconnect();
  }

  /**
   * Validate output against expected keywords and patterns
   */
  private validateOutput(
    output: string,
    payload: MochiQuery["payload"]
  ): {
    qualityScore: number;
    hallucinationRisk: number;
    foundKeywords: string[];
    missingKeywords: string[];
    violations: string[];
    patternMatched: boolean | null;
  } {
    const expected = payload.expectedKeywords ?? [];
    const forbidden = payload.forbiddenKeywords ?? [];

    // Check expected keywords
    const foundKeywords = expected.filter((k) =>
      output.toLowerCase().includes(k.toLowerCase())
    );
    const missingKeywords = expected.filter(
      (k) => !output.toLowerCase().includes(k.toLowerCase())
    );

    // Check forbidden keywords
    const violations = forbidden.filter((k) =>
      output.toLowerCase().includes(k.toLowerCase())
    );

    // Check pattern
    let patternMatched: boolean | null = null;
    if (payload.expectedPattern) {
      try {
        patternMatched = new RegExp(payload.expectedPattern).test(output);
      } catch {
        patternMatched = false;
      }
    }

    // Calculate quality score
    let qualityScore = 1.0;

    // Deduct for missing expected keywords
    if (expected.length > 0) {
      qualityScore *= foundKeywords.length / expected.length;
    }

    // Deduct for violations
    qualityScore *= Math.max(0, 1 - violations.length * 0.2);

    // Deduct for pattern mismatch
    if (patternMatched === false) {
      qualityScore *= 0.8;
    }

    // Calculate hallucination risk
    const hallucinationRisk = Math.min(1, violations.length * 0.25);

    return {
      qualityScore: Math.max(0, Math.min(1, qualityScore)),
      hallucinationRisk,
      foundKeywords,
      missingKeywords,
      violations,
      patternMatched,
    };
  }
}
