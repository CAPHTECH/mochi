#!/usr/bin/env tsx
/**
 * Simple test script for MCP client
 */

import { MCPClient } from "../src/mochi/assay/mcp-client.js";

async function main() {
  console.log("Testing MCP Client...");

  const client = new MCPClient("python3");

  console.log("Attempting ping...");
  const ok = await client.ping();
  console.log("Ping result:", ok);

  console.log("Disconnecting...");
  await client.disconnect();

  console.log("Done.");
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
