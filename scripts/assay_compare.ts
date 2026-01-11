#!/usr/bin/env tsx
/**
 * Mochi Comparison Evaluation Script
 *
 * Compares LLM performance with and without mochi context assistance.
 *
 * Usage:
 *   pnpm tsx scripts/assay_compare.ts
 *   pnpm tsx scripts/assay_compare.ts --model gpt-oss:120b
 *   pnpm tsx scripts/assay_compare.ts --output results.json
 *   pnpm tsx scripts/assay_compare.ts --include-mochi-only
 */

import { loadDataset } from "../vendor/assay-kit/packages/assay-kit/src/index.js";
import type { Dataset } from "../vendor/assay-kit/packages/assay-kit/src/types/dataset.js";
import {
  ComparisonAdapter,
  generateComparisonSummary,
  type ComparisonQuery,
  type ComparisonMetrics,
} from "../src/mochi/assay/comparison-adapter.js";
import * as fs from "fs/promises";
import * as path from "path";

interface ComparisonReport {
  timestamp: string;
  model: string;
  dataset: {
    name: string;
    version: string;
    totalQueries: number;
  };
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
  queries: Array<{
    queryId: string;
    category: string;
    baseline: {
      response: string;
      quality: number;
      hallucination: number;
      latencyMs: number;
      violations: string[];
    };
    assisted: {
      response: string;
      quality: number;
      hallucination: number;
      latencyMs: number;
      violations: string[];
    };
    improvement: {
      qualityDelta: number;
      hallucinationDelta: number;
    };
  }>;
}

function parseArgs(args: string[]): {
  model: string;
  outputPath: string;
  includeMochiOnly: boolean;
  ollamaEndpoint: string;
} {
  let model = "gpt-oss:120b";
  let outputPath = "output/comparison-results.json";
  let includeMochiOnly = false;
  let ollamaEndpoint = "http://localhost:11434";

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--model" && args[i + 1]) {
      model = args[++i];
    } else if (args[i] === "--output" && args[i + 1]) {
      outputPath = args[++i];
    } else if (args[i] === "--include-mochi-only") {
      includeMochiOnly = true;
    } else if (args[i] === "--ollama-endpoint" && args[i + 1]) {
      ollamaEndpoint = args[++i];
    }
  }

  return { model, outputPath, includeMochiOnly, ollamaEndpoint };
}

async function main() {
  const args = process.argv.slice(2);
  const { model, outputPath, includeMochiOnly, ollamaEndpoint } = parseArgs(args);

  console.log("=".repeat(70));
  console.log("Mochi Comparison Evaluation");
  console.log("=".repeat(70));
  console.log();
  console.log(`Model: ${model}`);
  console.log(`Ollama endpoint: ${ollamaEndpoint}`);
  console.log(`Include mochi-only: ${includeMochiOnly}`);
  console.log();

  // Load dataset
  const datasetPath = path.resolve("data/assay/mochi-eval.yaml");
  console.log(`Loading dataset: ${datasetPath}`);
  const dataset = (await loadDataset(datasetPath)) as Dataset<ComparisonQuery>;
  console.log(`Loaded ${dataset.queries.length} queries`);
  console.log();

  // Create adapter
  const adapter = new ComparisonAdapter({
    ollamaEndpoint,
    ollamaModel: model,
    includeMochiOnly,
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
  console.log("Running comparison evaluation...");
  console.log("-".repeat(70));

  const results: Array<{
    queryId: string;
    category: string;
    metrics: ComparisonMetrics;
  }> = [];

  const abortController = new AbortController();

  for (const query of dataset.queries) {
    try {
      const category = (query.metadata?.category as string) || "unknown";
      console.log(`\n[${query.id}] ${query.text}`);
      console.log(`  Category: ${category}`);

      const metrics = await adapter.execute(query, {
        signal: abortController.signal,
        runIndex: 0,
      });

      results.push({
        queryId: query.id,
        category,
        metrics,
      });

      // Print quick summary
      const baseQ = metrics.baseline.keywordScore.toFixed(2);
      const assistQ = metrics.assisted.keywordScore.toFixed(2);
      const delta = metrics.improvement.qualityDelta > 0 ? "+" : "";
      console.log(
        `  Baseline: ${baseQ} → Assisted: ${assistQ} (${delta}${(metrics.improvement.qualityDelta * 100).toFixed(0)}%)`
      );

      if (metrics.baseline.violations.length > 0) {
        console.log(`  Baseline violations: ${metrics.baseline.violations.join(", ")}`);
      }
      if (metrics.assisted.violations.length > 0) {
        console.log(`  Assisted violations: ${metrics.assisted.violations.join(", ")}`);
      }
    } catch (error) {
      console.error(`  ERROR: ${error instanceof Error ? error.message : error}`);
    }
  }

  // Generate summary
  const summary = generateComparisonSummary(results, model);

  // Build full report
  const report: ComparisonReport = {
    timestamp: summary.timestamp,
    model: summary.model,
    dataset: {
      name: dataset.name,
      version: dataset.version,
      totalQueries: dataset.queries.length,
    },
    summary: summary.summary,
    byCategory: summary.byCategory,
    queries: results.map((r) => ({
      queryId: r.queryId,
      category: r.category,
      baseline: {
        response: r.metrics.baseline.response,
        quality: r.metrics.baseline.keywordScore,
        hallucination: r.metrics.baseline.hallucinationScore,
        latencyMs: r.metrics.baseline.latencyMs,
        violations: r.metrics.baseline.violations,
      },
      assisted: {
        response: r.metrics.assisted.response,
        quality: r.metrics.assisted.keywordScore,
        hallucination: r.metrics.assisted.hallucinationScore,
        latencyMs: r.metrics.assisted.latencyMs,
        violations: r.metrics.assisted.violations,
      },
      improvement: {
        qualityDelta: r.metrics.improvement.qualityDelta,
        hallucinationDelta: r.metrics.improvement.hallucinationDelta,
      },
    })),
  };

  // Print summary
  console.log();
  console.log("=".repeat(70));
  console.log("COMPARISON RESULTS");
  console.log("=".repeat(70));
  console.log();

  console.log("Overall Summary:");
  console.log(
    `  Baseline Avg Quality:     ${summary.summary.baselineAvgQuality.toFixed(3)}`
  );
  console.log(
    `  Assisted Avg Quality:     ${summary.summary.assistedAvgQuality.toFixed(3)}`
  );
  console.log(
    `  Quality Improvement:      ${summary.summary.improvementPercent >= 0 ? "+" : ""}${summary.summary.improvementPercent.toFixed(1)}%`
  );
  console.log();
  console.log(
    `  Baseline Avg Hallucination: ${summary.summary.baselineAvgHallucination.toFixed(3)}`
  );
  console.log(
    `  Assisted Avg Hallucination: ${summary.summary.assistedAvgHallucination.toFixed(3)}`
  );
  console.log(
    `  Hallucination Reduction:    ${summary.summary.hallucinationReductionPercent >= 0 ? "-" : "+"}${Math.abs(summary.summary.hallucinationReductionPercent).toFixed(1)}%`
  );
  console.log();

  console.log("By Category:");
  for (const [cat, data] of Object.entries(summary.byCategory)) {
    const delta = data.improvement >= 0 ? "+" : "";
    console.log(
      `  ${cat}: ${data.baselineAvgQuality.toFixed(2)} → ${data.assistedAvgQuality.toFixed(2)} (${delta}${data.improvement.toFixed(0)}%)`
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
    summary.summary.improvementPercent >= 50 &&
    summary.summary.hallucinationReductionPercent >= 50;

  if (success) {
    console.log("\nResult: PASS (mochi provides significant improvement)");
    process.exit(0);
  } else if (summary.summary.improvementPercent > 0) {
    console.log("\nResult: PARTIAL (mochi provides some improvement)");
    process.exit(0);
  } else {
    console.log("\nResult: NEEDS IMPROVEMENT");
    process.exit(1);
  }
}

main().catch((error) => {
  console.error("Comparison evaluation failed:", error);
  process.exit(1);
});
