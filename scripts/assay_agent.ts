#!/usr/bin/env tsx
/**
 * Mochi Agent Evaluation Script
 *
 * Evaluates LLM as an agent with tool-calling capabilities.
 * Measures tool decision accuracy, context utilization, and code quality.
 *
 * Usage:
 *   pnpm tsx scripts/assay_agent.ts
 *   pnpm tsx scripts/assay_agent.ts --model gpt-oss:120b
 *   pnpm tsx scripts/assay_agent.ts --output results.json
 *   pnpm tsx scripts/assay_agent.ts --category llm-assist
 */

import { loadDataset } from "../vendor/assay-kit/packages/assay-kit/src/index.js";
import type { Dataset } from "../vendor/assay-kit/packages/assay-kit/src/types/dataset.js";
import {
  AgentAdapter,
  generateAgentSummary,
  type AgentQuery,
  type AgentMetrics,
} from "../src/mochi/assay/agent-adapter.js";
import * as fs from "fs/promises";
import * as path from "path";

interface AgentReport {
  timestamp: string;
  model: string;
  dataset: {
    name: string;
    version: string;
    totalQueries: number;
    evaluatedQueries: number;
  };
  summary: {
    toolDecisionAccuracy: number;
    toolCallRate: number;
    avgContextUtilization: number;
    usedContextRate: number;
    avgCodeQuality: number;
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
  queries: Array<{
    queryId: string;
    category: string;
    response: string;
    latencyMs: number;
    toolDecision: {
      calledTool: boolean;
      decisionCorrect: boolean;
      toolCalled?: string;
    };
    contextUtilization: {
      receivedContext: boolean;
      usedContext: boolean;
      utilizationScore: number;
    };
    codeQuality: {
      keywordScore: number;
      hallucinationScore: number;
      foundKeywords: string[];
      missingKeywords: string[];
      violations: string[];
    };
  }>;
}

function parseArgs(args: string[]): {
  model: string;
  outputPath: string;
  ollamaEndpoint: string;
  category: string | null;
} {
  let model = "gpt-oss:120b";
  let outputPath = "output/agent-results.json";
  let ollamaEndpoint = "http://localhost:11434";
  let category: string | null = null;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--model" && args[i + 1]) {
      model = args[++i];
    } else if (args[i] === "--output" && args[i + 1]) {
      outputPath = args[++i];
    } else if (args[i] === "--ollama-endpoint" && args[i + 1]) {
      ollamaEndpoint = args[++i];
    } else if (args[i] === "--category" && args[i + 1]) {
      category = args[++i];
    }
  }

  return { model, outputPath, ollamaEndpoint, category };
}

async function main() {
  const args = process.argv.slice(2);
  const { model, outputPath, ollamaEndpoint, category } = parseArgs(args);

  console.log("=".repeat(70));
  console.log("Mochi Agent Evaluation");
  console.log("=".repeat(70));
  console.log();
  console.log(`Model: ${model}`);
  console.log(`Ollama endpoint: ${ollamaEndpoint}`);
  if (category) {
    console.log(`Category filter: ${category}`);
  }
  console.log();

  // Load dataset
  const datasetPath = path.resolve("data/assay/mochi-eval.yaml");
  console.log(`Loading dataset: ${datasetPath}`);
  const dataset = (await loadDataset(datasetPath)) as Dataset<AgentQuery>;

  // Filter queries with shouldCallTool metadata
  let queries = dataset.queries.filter(
    (q) => q.metadata?.shouldCallTool !== undefined
  );

  // Filter by category if specified
  if (category) {
    queries = queries.filter((q) => q.metadata?.category === category);
  }

  console.log(
    `Found ${queries.length} queries with agent metadata (out of ${dataset.queries.length} total)`
  );
  console.log();

  if (queries.length === 0) {
    console.error("No queries found with shouldCallTool metadata.");
    console.error(
      "Make sure the dataset has metadata.shouldCallTool defined for agent evaluation."
    );
    process.exit(1);
  }

  // Create adapter
  const adapter = new AgentAdapter({
    ollamaEndpoint,
    ollamaModel: model,
    maxIterations: 3,
  });

  // Warmup
  console.log("Warming up...");
  try {
    await adapter.warmup(dataset);
  } catch (error) {
    console.error("Warmup failed:", error);
    console.error("\nMake sure:");
    console.error("  1. Ollama is running: ollama serve");
    console.error(`  2. Model is available: ollama pull ${model}`);
    console.error("  3. Python environment has mochi installed");
    process.exit(1);
  }
  console.log();

  // Run evaluation
  console.log("Running agent evaluation...");
  console.log("-".repeat(70));

  const results: Array<{
    queryId: string;
    category: string;
    metrics: AgentMetrics;
  }> = [];

  const abortController = new AbortController();

  for (const query of queries) {
    try {
      const cat = (query.metadata?.category as string) || "unknown";
      const expectedTool = query.metadata?.expectedTool || "any";
      const shouldCall = query.metadata?.shouldCallTool;

      console.log(`\n[${query.id}] ${query.text}`);
      console.log(`  Category: ${cat}`);
      console.log(`  Should call tool: ${shouldCall} (expected: ${expectedTool})`);

      const metrics = await adapter.execute(query, {
        signal: abortController.signal,
        runIndex: 0,
      });

      results.push({
        queryId: query.id,
        category: cat,
        metrics,
      });

      // Print quick summary
      const decision = metrics.toolDecision.decisionCorrect ? "CORRECT" : "WRONG";
      const toolInfo = metrics.toolDecision.calledTool
        ? `called ${metrics.toolDecision.toolCalled}`
        : "no tool call";
      console.log(`  Tool decision: ${decision} (${toolInfo})`);

      if (metrics.contextUtilization.receivedContext) {
        console.log(
          `  Context utilization: ${(metrics.contextUtilization.utilizationScore * 100).toFixed(0)}%`
        );
      }

      const quality = metrics.codeQuality.keywordScore.toFixed(2);
      const hallucination = metrics.codeQuality.hallucinationScore.toFixed(2);
      console.log(`  Code quality: ${quality} (hallucination risk: ${hallucination})`);

      if (metrics.codeQuality.violations.length > 0) {
        console.log(`  Violations: ${metrics.codeQuality.violations.join(", ")}`);
      }
    } catch (error) {
      console.error(`  ERROR: ${error instanceof Error ? error.message : error}`);
    }
  }

  // Generate summary
  const summary = generateAgentSummary(results, model);

  // Build full report
  const report: AgentReport = {
    timestamp: summary.timestamp,
    model: summary.model,
    dataset: {
      name: dataset.name,
      version: dataset.version,
      totalQueries: dataset.queries.length,
      evaluatedQueries: queries.length,
    },
    summary: {
      toolDecisionAccuracy: summary.toolDecision.accuracy,
      toolCallRate: summary.toolDecision.callRate,
      avgContextUtilization: summary.contextUtilization.avgUtilizationScore,
      usedContextRate: summary.contextUtilization.usedContextRate,
      avgCodeQuality: summary.codeQuality.avgKeywordScore,
      avgHallucinationScore: summary.codeQuality.avgHallucinationScore,
    },
    byCategory: summary.byCategory,
    queries: results.map((r) => ({
      queryId: r.queryId,
      category: r.category,
      response: r.metrics.response,
      latencyMs: r.metrics.latencyMs,
      toolDecision: {
        calledTool: r.metrics.toolDecision.calledTool,
        decisionCorrect: r.metrics.toolDecision.decisionCorrect,
        toolCalled: r.metrics.toolDecision.toolCalled,
      },
      contextUtilization: {
        receivedContext: r.metrics.contextUtilization.receivedContext,
        usedContext: r.metrics.contextUtilization.usedContext,
        utilizationScore: r.metrics.contextUtilization.utilizationScore,
      },
      codeQuality: {
        keywordScore: r.metrics.codeQuality.keywordScore,
        hallucinationScore: r.metrics.codeQuality.hallucinationScore,
        foundKeywords: r.metrics.codeQuality.foundKeywords,
        missingKeywords: r.metrics.codeQuality.missingKeywords,
        violations: r.metrics.codeQuality.violations,
      },
    })),
  };

  // Print summary
  console.log();
  console.log("=".repeat(70));
  console.log("AGENT EVALUATION RESULTS");
  console.log("=".repeat(70));
  console.log();

  console.log("Tool Decision:");
  console.log(
    `  Accuracy:     ${(summary.toolDecision.accuracy * 100).toFixed(1)}%`
  );
  console.log(
    `  Call Rate:    ${(summary.toolDecision.callRate * 100).toFixed(1)}%`
  );
  console.log(
    `  Correct:      ${summary.toolDecision.correctCalls}/${summary.totalQueries}`
  );
  console.log();

  console.log("Context Utilization:");
  console.log(
    `  Avg Utilization: ${(summary.contextUtilization.avgUtilizationScore * 100).toFixed(1)}%`
  );
  console.log(
    `  Used Context:    ${(summary.contextUtilization.usedContextRate * 100).toFixed(1)}%`
  );
  console.log();

  console.log("Code Quality:");
  console.log(
    `  Avg Keyword Score:      ${summary.codeQuality.avgKeywordScore.toFixed(3)}`
  );
  console.log(
    `  Avg Hallucination Risk: ${summary.codeQuality.avgHallucinationScore.toFixed(3)}`
  );
  console.log();

  console.log("By Category:");
  for (const [cat, data] of Object.entries(summary.byCategory)) {
    console.log(
      `  ${cat}: accuracy=${(data.toolAccuracy * 100).toFixed(0)}%, quality=${data.avgQuality.toFixed(2)}`
    );
  }
  console.log();

  // Save report
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, JSON.stringify(report, null, 2));
  console.log(`Report saved to: ${outputPath}`);

  // Cleanup
  await adapter.stop();

  // Exit with appropriate code
  const success =
    summary.toolDecision.accuracy >= 0.7 &&
    summary.codeQuality.avgKeywordScore >= 0.6 &&
    summary.codeQuality.avgHallucinationScore <= 0.3;

  if (success) {
    console.log("\nResult: PASS (agent effectively uses mochi tools)");
    process.exit(0);
  } else if (summary.toolDecision.accuracy >= 0.5) {
    console.log("\nResult: PARTIAL (agent sometimes uses mochi tools correctly)");
    process.exit(0);
  } else {
    console.log("\nResult: NEEDS IMPROVEMENT (agent tool usage needs work)");
    process.exit(1);
  }
}

main().catch((error) => {
  console.error("Agent evaluation failed:", error);
  process.exit(1);
});
