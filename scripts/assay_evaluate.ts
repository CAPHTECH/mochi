#!/usr/bin/env tsx
/**
 * Mochi evaluation script using assay-kit framework.
 *
 * Usage:
 *   pnpm tsx scripts/assay_evaluate.ts
 *   pnpm tsx scripts/assay_evaluate.ts --output results.json
 */

import { loadDataset, Runner } from "../vendor/assay-kit/packages/assay-kit/src/index.js";
import type { Dataset } from "../vendor/assay-kit/packages/assay-kit/src/types/dataset.js";
import { MochiAdapter } from "../src/mochi/assay/adapter.js";
import type { MochiQuery, MochiMetrics } from "../src/mochi/assay/adapter.js";
import * as fs from "fs/promises";
import * as path from "path";

interface EvaluationSummary {
  timestamp: string;
  dataset: {
    name: string;
    version: string;
    totalQueries: number;
  };
  overall: {
    successCount: number;
    errorCount: number;
    avgLatencyMs: number;
    avgConfidence: number;
    avgQualityScore: number;
    avgHallucinationRisk: number;
  };
  byCategory: Record<
    string,
    {
      count: number;
      successCount: number;
      avgQualityScore: number;
    }
  >;
  queries: Array<{
    queryId: string;
    status: "success" | "error" | "timeout";
    metrics?: MochiMetrics;
    error?: string;
  }>;
}

async function main() {
  const args = process.argv.slice(2);
  const outputPath = args.includes("--output")
    ? args[args.indexOf("--output") + 1]
    : "output/assay-results.json";

  console.log("=" .repeat(70));
  console.log("Mochi Evaluation with assay-kit");
  console.log("=" .repeat(70));
  console.log();

  // Load dataset
  const datasetPath = path.resolve("data/assay/mochi-eval.yaml");
  console.log(`Loading dataset: ${datasetPath}`);
  const dataset = await loadDataset(datasetPath) as Dataset<MochiQuery>;
  console.log(`Loaded ${dataset.queries.length} queries`);
  console.log();

  // Create adapter
  const adapter = new MochiAdapter({
    pythonCommand: "python3",
  });

  // Create runner
  const runner = new Runner({
    adapter,
    warmupRuns: 1, // Preload model during warmup
    maxRetries: 1,
    concurrency: 1, // Sequential execution for mochi
    timeoutMs: 180000, // 3 minutes timeout for inference
  });

  // Run evaluation
  console.log("Running evaluation...");
  console.log();

  const results = await runner.evaluate(dataset);

  // Process results
  const summary: EvaluationSummary = {
    timestamp: new Date().toISOString(),
    dataset: {
      name: dataset.name,
      version: dataset.version,
      totalQueries: dataset.queries.length,
    },
    overall: {
      successCount: 0,
      errorCount: 0,
      avgLatencyMs: 0,
      avgConfidence: 0,
      avgQualityScore: 0,
      avgHallucinationRisk: 0,
    },
    byCategory: {},
    queries: [],
  };

  let totalLatency = 0;
  let totalConfidence = 0;
  let totalQuality = 0;
  let totalHallucination = 0;
  let successCount = 0;

  for (const q of results.queries) {
    const queryDef = dataset.queries.find((qd) => qd.id === q.queryId);
    const category = (queryDef?.metadata?.category as string) || "unknown";

    if (!summary.byCategory[category]) {
      summary.byCategory[category] = {
        count: 0,
        successCount: 0,
        avgQualityScore: 0,
      };
    }
    summary.byCategory[category].count++;

    if (q.status === "success" && q.metrics) {
      const metrics = q.metrics as MochiMetrics;
      successCount++;
      summary.overall.successCount++;
      summary.byCategory[category].successCount++;

      totalLatency += metrics.latencyMs;
      totalConfidence += metrics.confidence;
      totalQuality += metrics.qualityScore;
      totalHallucination += metrics.hallucinationRisk;

      summary.queries.push({
        queryId: q.queryId,
        status: "success",
        metrics,
      });
    } else {
      summary.overall.errorCount++;
      const err = q.error as unknown;
      const errorMsg = err instanceof Error
        ? err.message
        : typeof err === "string"
          ? err
          : "Unknown error";
      summary.queries.push({
        queryId: q.queryId,
        status: q.status as "error" | "timeout",
        error: errorMsg,
      });
    }
  }

  // Calculate averages
  if (successCount > 0) {
    summary.overall.avgLatencyMs = Math.round(totalLatency / successCount);
    summary.overall.avgConfidence = totalConfidence / successCount;
    summary.overall.avgQualityScore = totalQuality / successCount;
    summary.overall.avgHallucinationRisk = totalHallucination / successCount;
  }

  // Calculate category averages
  for (const [cat, data] of Object.entries(summary.byCategory)) {
    const catQueries = summary.queries.filter(
      (q) =>
        q.status === "success" &&
        dataset.queries.find((qd) => qd.id === q.queryId)?.metadata
          ?.category === cat
    );
    if (catQueries.length > 0) {
      data.avgQualityScore =
        catQueries.reduce(
          (sum, q) => sum + (q.metrics?.qualityScore || 0),
          0
        ) / catQueries.length;
    }
  }

  // Print results
  console.log("=" .repeat(70));
  console.log("RESULTS");
  console.log("=" .repeat(70));
  console.log();

  console.log(`Success Rate: ${summary.overall.successCount}/${dataset.queries.length} (${((summary.overall.successCount / dataset.queries.length) * 100).toFixed(1)}%)`);
  console.log(`Avg Latency: ${summary.overall.avgLatencyMs}ms`);
  console.log(`Avg Confidence: ${summary.overall.avgConfidence.toFixed(3)}`);
  console.log(`Avg Quality Score: ${summary.overall.avgQualityScore.toFixed(3)}`);
  console.log(`Avg Hallucination Risk: ${summary.overall.avgHallucinationRisk.toFixed(3)}`);
  console.log();

  console.log("By Category:");
  for (const [cat, data] of Object.entries(summary.byCategory)) {
    console.log(`  ${cat}: ${data.successCount}/${data.count} success, avgQuality=${data.avgQualityScore.toFixed(3)}`);
  }
  console.log();

  console.log("Individual Results:");
  for (const q of summary.queries) {
    if (q.status === "success" && q.metrics) {
      const status =
        q.metrics.qualityScore >= 0.8
          ? "PASS"
          : q.metrics.qualityScore >= 0.5
            ? "WARN"
            : "FAIL";
      console.log(`  [${status}] ${q.queryId}`);
      console.log(`        confidence=${q.metrics.confidence.toFixed(2)}, quality=${q.metrics.qualityScore.toFixed(2)}, hallucination=${q.metrics.hallucinationRisk.toFixed(2)}`);
      if (q.metrics.extras.missingKeywords.length > 0) {
        console.log(`        missing: ${q.metrics.extras.missingKeywords.join(", ")}`);
      }
      if (q.metrics.extras.violations.length > 0) {
        console.log(`        violations: ${q.metrics.extras.violations.join(", ")}`);
      }
    } else {
      console.log(`  [ERROR] ${q.queryId}: ${q.error}`);
    }
  }
  console.log();

  // Save results
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, JSON.stringify(summary, null, 2));
  console.log(`Results saved to: ${outputPath}`);

  // Cleanup
  await adapter.stop();

  // Exit with appropriate code
  const passRate = summary.overall.successCount / dataset.queries.length;
  const avgQuality = summary.overall.avgQualityScore;

  if (passRate >= 0.8 && avgQuality >= 0.7) {
    console.log("\nOverall: PASS");
    process.exit(0);
  } else {
    console.log("\nOverall: NEEDS IMPROVEMENT");
    process.exit(1);
  }
}

main().catch((error) => {
  console.error("Evaluation failed:", error);
  process.exit(1);
});
