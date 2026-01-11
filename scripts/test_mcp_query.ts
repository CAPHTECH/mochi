#!/usr/bin/env tsx
/**
 * Test MCP client with domain_query call
 */

import { MCPClient } from "../src/mochi/assay/mcp-client.js";

async function main() {
  console.log("Testing MCP Client domain_query...");
  console.log("This will load the model (may take a few minutes)...\n");

  const client = new MCPClient("python3");

  try {
    console.log("Calling domain_query...");
    const result = await client.call("domain_query", {
      instruction: "Fill in the typescript code",
      input: "// File: src/test.ts\nconst users = await db.",
      context: `// Available methods on DuckDBClient:
//   all<T>(sql: string, params?: any[]): Promise<T[]>
//   run(sql: string, params?: any[]): Promise<void>`,
      validate: true,
      mode: "auto",
    });

    console.log("\n=== Result ===");
    console.log("Response:", result.response);
    console.log("Confidence:", result.confidence);
    console.log("Inference time:", result.inference_time_ms, "ms");
    console.log("Tokens generated:", result.tokens_generated);
    console.log("Mode used:", result.mode_used);
    console.log("Retried:", result.retried);
    if (result.validation) {
      console.log("Validation:", result.validation);
    }
  } catch (error) {
    console.error("Error:", error);
  } finally {
    console.log("\nDisconnecting...");
    await client.disconnect();
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
