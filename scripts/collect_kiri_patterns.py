#!/usr/bin/env python3
"""Collect kiri-specific patterns for Project Adapter training.

Creates curated training examples based on patterns found in kiri codebase:
1. Singleton Registry Pattern
2. Discriminated Union Types
3. Weighted Profile Selection
4. Configuration Validation
5. Path-based Multipliers
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Singleton Registry Patterns
SINGLETON_REGISTRY_PATTERNS = [
    {
        "instruction": "Implement a singleton registry pattern for managing language analyzers",
        "input": """interface LanguageAnalyzer {
  language: string;
  analyze(context: AnalysisContext): Promise<AnalysisResult>;
  dispose?(): Promise<void>;
}""",
        "output": """export class LanguageRegistry {
  private static instance: LanguageRegistry | null = null;
  private readonly analyzers = new Map<string, LanguageAnalyzer>();

  private constructor() {}

  static getInstance(): LanguageRegistry {
    if (!LanguageRegistry.instance) {
      LanguageRegistry.instance = new LanguageRegistry();
    }
    return LanguageRegistry.instance;
  }

  register(analyzer: LanguageAnalyzer): void {
    if (this.analyzers.has(analyzer.language)) {
      throw new Error(`Analyzer already registered for language: ${analyzer.language}`);
    }
    this.analyzers.set(analyzer.language, analyzer);
  }

  async analyze(language: string | null, context: AnalysisContext): Promise<AnalysisResult> {
    if (!language) return emptyResult();
    const analyzer = this.analyzers.get(language);
    if (!analyzer) return emptyResult();
    return analyzer.analyze(context);
  }

  async cleanup(): Promise<void> {
    const disposePromises: Promise<void>[] = [];
    for (const analyzer of this.analyzers.values()) {
      if (analyzer.dispose) {
        disposePromises.push(analyzer.dispose().catch(console.error));
      }
    }
    await Promise.allSettled(disposePromises);
    this.analyzers.clear();
  }
}""",
    },
    {
        "instruction": "Create a singleton plugin manager with register and get methods",
        "input": """interface Plugin {
  name: string;
  init(): Promise<void>;
  destroy?(): Promise<void>;
}""",
        "output": """export class PluginManager {
  private static instance: PluginManager | null = null;
  private readonly plugins = new Map<string, Plugin>();

  private constructor() {}

  static getInstance(): PluginManager {
    if (!PluginManager.instance) {
      PluginManager.instance = new PluginManager();
    }
    return PluginManager.instance;
  }

  async register(plugin: Plugin): Promise<void> {
    if (this.plugins.has(plugin.name)) {
      throw new Error(`Plugin already registered: ${plugin.name}`);
    }
    await plugin.init();
    this.plugins.set(plugin.name, plugin);
  }

  get(name: string): Plugin | undefined {
    return this.plugins.get(name);
  }

  async destroyAll(): Promise<void> {
    for (const plugin of this.plugins.values()) {
      if (plugin.destroy) {
        await plugin.destroy().catch(console.error);
      }
    }
    this.plugins.clear();
  }
}""",
    },
    {
        "instruction": "Implement a singleton service locator pattern",
        "input": """interface Service {
  name: string;
  start(): Promise<void>;
  stop(): Promise<void>;
}""",
        "output": """export class ServiceLocator {
  private static instance: ServiceLocator | null = null;
  private readonly services = new Map<string, Service>();

  private constructor() {}

  static getInstance(): ServiceLocator {
    if (!ServiceLocator.instance) {
      ServiceLocator.instance = new ServiceLocator();
    }
    return ServiceLocator.instance;
  }

  async register(service: Service): Promise<void> {
    if (this.services.has(service.name)) {
      throw new Error(`Service already registered: ${service.name}`);
    }
    await service.start();
    this.services.set(service.name, service);
  }

  get<T extends Service>(name: string): T | undefined {
    return this.services.get(name) as T | undefined;
  }

  async shutdown(): Promise<void> {
    const stopPromises = Array.from(this.services.values()).map((s) =>
      s.stop().catch((err) => console.error(`Failed to stop ${s.name}:`, err))
    );
    await Promise.allSettled(stopPromises);
    this.services.clear();
  }
}""",
    },
]

# Discriminated Union Patterns
DISCRIMINATED_UNION_PATTERNS = [
    {
        "instruction": "Create discriminated union types for content overlay changes",
        "input": """// Need types for:
// - AddContentOverlay with type: "add" and content: string
// - RemoveContentOverlay with type: "remove"
// - ContentOverlayChange as union of both""",
        "output": """export interface AddContentOverlay {
  type: "add";
  content: string;
}

export interface RemoveContentOverlay {
  type: "remove";
}

export type ContentOverlayChange = AddContentOverlay | RemoveContentOverlay;

export function isAddOverlay(change: ContentOverlayChange): change is AddContentOverlay {
  return change.type === "add";
}

export function isRemoveOverlay(change: ContentOverlayChange): change is RemoveContentOverlay {
  return change.type === "remove";
}""",
    },
    {
        "instruction": "Create discriminated union for API response states",
        "input": """// Need types for:
// - Loading state with type: "loading"
// - Success state with type: "success" and data
// - Error state with type: "error" and message""",
        "output": """export interface LoadingState {
  type: "loading";
}

export interface SuccessState<T> {
  type: "success";
  data: T;
}

export interface ErrorState {
  type: "error";
  message: string;
  code?: number;
}

export type AsyncState<T> = LoadingState | SuccessState<T> | ErrorState;

export function isLoading<T>(state: AsyncState<T>): state is LoadingState {
  return state.type === "loading";
}

export function isSuccess<T>(state: AsyncState<T>): state is SuccessState<T> {
  return state.type === "success";
}

export function isError<T>(state: AsyncState<T>): state is ErrorState {
  return state.type === "error";
}""",
    },
    {
        "instruction": "Create discriminated union for file system events",
        "input": """// Need types for file events:
// - Created with type: "created" and path
// - Modified with type: "modified" and path, content
// - Deleted with type: "deleted" and path""",
        "output": """export interface FileCreatedEvent {
  type: "created";
  path: string;
  timestamp: number;
}

export interface FileModifiedEvent {
  type: "modified";
  path: string;
  content: string;
  timestamp: number;
}

export interface FileDeletedEvent {
  type: "deleted";
  path: string;
  timestamp: number;
}

export type FileEvent = FileCreatedEvent | FileModifiedEvent | FileDeletedEvent;

export function handleFileEvent(event: FileEvent): void {
  switch (event.type) {
    case "created":
      console.log(`File created: ${event.path}`);
      break;
    case "modified":
      console.log(`File modified: ${event.path}, size: ${event.content.length}`);
      break;
    case "deleted":
      console.log(`File deleted: ${event.path}`);
      break;
  }
}""",
    },
    {
        "instruction": "Create discriminated union for message types in a chat system",
        "input": """// Need types for chat messages:
// - TextMessage with type: "text" and content
// - ImageMessage with type: "image" and url, caption
// - SystemMessage with type: "system" and action""",
        "output": """export interface TextMessage {
  type: "text";
  id: string;
  sender: string;
  content: string;
  timestamp: number;
}

export interface ImageMessage {
  type: "image";
  id: string;
  sender: string;
  url: string;
  caption?: string;
  timestamp: number;
}

export interface SystemMessage {
  type: "system";
  id: string;
  action: "join" | "leave" | "rename";
  details: string;
  timestamp: number;
}

export type ChatMessage = TextMessage | ImageMessage | SystemMessage;

export function renderMessage(message: ChatMessage): string {
  switch (message.type) {
    case "text":
      return `${message.sender}: ${message.content}`;
    case "image":
      return `${message.sender} shared an image: ${message.caption || "No caption"}`;
    case "system":
      return `[System] ${message.details}`;
  }
}""",
    },
]

# Weighted Profile Selection Patterns
WEIGHTED_SCORING_PATTERNS = [
    {
        "instruction": "Implement a weighted profile selection function based on keyword matching",
        "input": """interface ProfilePattern {
  profile: string;
  keywords: string[];
  weight: number;
}

const patterns: ProfilePattern[] = [
  { profile: "testfail", keywords: ["test fail", "failing test"], weight: 10 },
  { profile: "typeerror", keywords: ["type error", "type mismatch"], weight: 11 },
];""",
        "output": """interface CompiledPattern {
  profile: string;
  regex: RegExp;
  weight: number;
}

const COMPILED_PATTERNS: CompiledPattern[] = patterns.map((pattern) => {
  const escapedKeywords = pattern.keywords.map((k) =>
    k.toLowerCase().replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&")
  );
  const regexPattern = escapedKeywords.join("|");
  return {
    profile: pattern.profile,
    regex: new RegExp(regexPattern, "gi"),
    weight: pattern.weight,
  };
});

export function selectProfileFromQuery(
  query: string,
  fallback: string = "default"
): string {
  const normalizedQuery = query.toLowerCase().trim();
  if (normalizedQuery.length === 0) return fallback;

  let bestMatch = fallback;
  let highestScore = 0;

  for (const pattern of COMPILED_PATTERNS) {
    const matches = normalizedQuery.match(pattern.regex);
    const matchCount = matches ? matches.length : 0;
    if (matchCount > 0) {
      const score = matchCount * pattern.weight;
      if (score > highestScore) {
        highestScore = score;
        bestMatch = pattern.profile;
      }
    }
  }
  return bestMatch;
}""",
    },
    {
        "instruction": "Create a tag-based content scorer with weighted matching",
        "input": """interface TagWeight {
  tag: string;
  weight: number;
}

const tagWeights: TagWeight[] = [
  { tag: "important", weight: 5 },
  { tag: "urgent", weight: 10 },
  { tag: "low-priority", weight: 1 },
];""",
        "output": """interface ScoredContent {
  content: string;
  tags: string[];
  score: number;
}

export function scoreContent(content: string, tags: string[]): ScoredContent {
  let totalScore = 0;

  for (const tag of tags) {
    const tagLower = tag.toLowerCase();
    const weightEntry = tagWeights.find((tw) => tw.tag.toLowerCase() === tagLower);
    if (weightEntry) {
      totalScore += weightEntry.weight;
    } else {
      totalScore += 1; // Default weight for unknown tags
    }
  }

  return {
    content,
    tags,
    score: totalScore,
  };
}

export function sortByScore(items: ScoredContent[]): ScoredContent[] {
  return [...items].sort((a, b) => b.score - a.score);
}

export function filterByMinScore(
  items: ScoredContent[],
  minScore: number
): ScoredContent[] {
  return items.filter((item) => item.score >= minScore);
}""",
    },
    {
        "instruction": "Implement a priority queue with weighted items",
        "input": """interface WeightedItem<T> {
  item: T;
  priority: number;
}""",
        "output": """export class PriorityQueue<T> {
  private items: WeightedItem<T>[] = [];

  enqueue(item: T, priority: number): void {
    const weightedItem: WeightedItem<T> = { item, priority };
    const insertIndex = this.items.findIndex((i) => i.priority < priority);
    if (insertIndex === -1) {
      this.items.push(weightedItem);
    } else {
      this.items.splice(insertIndex, 0, weightedItem);
    }
  }

  dequeue(): T | undefined {
    const weightedItem = this.items.shift();
    return weightedItem?.item;
  }

  peek(): T | undefined {
    return this.items[0]?.item;
  }

  get size(): number {
    return this.items.length;
  }

  isEmpty(): boolean {
    return this.items.length === 0;
  }

  clear(): void {
    this.items = [];
  }
}""",
    },
]

# Configuration Validation Patterns
CONFIG_VALIDATION_PATTERNS = [
    {
        "instruction": "Add validation for scoring weights configuration with backward compatibility",
        "input": """interface ScoringWeights {
  textMatch: number;
  pathMatch: number;
  graphInbound?: number; // new field, optional for backward compat
}""",
        "output": """export function validateWeights(weights: unknown, profileName: string): ScoringWeights {
  if (typeof weights !== "object" || weights === null) {
    throw new Error(`Profile '${profileName}' must be an object`);
  }

  const obj = weights as Record<string, unknown>;

  // Backward compatibility: provide defaults for new fields
  if (obj.graphInbound === undefined) {
    obj.graphInbound = 0.5;
  }

  // Validate required fields
  const required = ["textMatch", "pathMatch"];
  for (const key of required) {
    const value = obj[key];
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
      throw new Error(
        `Profile '${profileName}' has invalid ${key}: ${String(value)}. Must be a non-negative finite number.`
      );
    }
  }

  // Validate optional fields if present
  if (obj.graphInbound !== undefined) {
    if (typeof obj.graphInbound !== "number" || !Number.isFinite(obj.graphInbound) || obj.graphInbound < 0) {
      throw new Error(`Profile '${profileName}' has invalid graphInbound`);
    }
  }

  return obj as ScoringWeights;
}""",
    },
    {
        "instruction": "Create a configuration loader with validation and fallback defaults",
        "input": """interface AppConfig {
  port: number;
  host: string;
  debug?: boolean;
  timeout?: number;
}""",
        "output": """const DEFAULT_CONFIG: Required<AppConfig> = {
  port: 3000,
  host: "localhost",
  debug: false,
  timeout: 30000,
};

export function loadConfig(config: unknown): Required<AppConfig> {
  if (typeof config !== "object" || config === null) {
    console.warn("Invalid config, using defaults");
    return { ...DEFAULT_CONFIG };
  }

  const obj = config as Record<string, unknown>;
  const result: Required<AppConfig> = { ...DEFAULT_CONFIG };

  // Validate and set port
  if (typeof obj.port === "number" && Number.isInteger(obj.port) && obj.port > 0 && obj.port < 65536) {
    result.port = obj.port;
  } else if (obj.port !== undefined) {
    console.warn(`Invalid port: ${obj.port}, using default: ${DEFAULT_CONFIG.port}`);
  }

  // Validate and set host
  if (typeof obj.host === "string" && obj.host.length > 0) {
    result.host = obj.host;
  } else if (obj.host !== undefined) {
    console.warn(`Invalid host: ${obj.host}, using default: ${DEFAULT_CONFIG.host}`);
  }

  // Validate and set debug
  if (typeof obj.debug === "boolean") {
    result.debug = obj.debug;
  }

  // Validate and set timeout
  if (typeof obj.timeout === "number" && obj.timeout > 0) {
    result.timeout = obj.timeout;
  }

  return result;
}""",
    },
    {
        "instruction": "Implement environment variable configuration with type coercion",
        "input": """interface EnvConfig {
  apiKey: string;
  maxRetries: number;
  enableCache: boolean;
  cacheTimeout: number;
}""",
        "output": """function readEnvString(name: string, fallback: string): string {
  const value = process.env[name];
  return value !== undefined ? value : fallback;
}

function readEnvNumber(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined) return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function readEnvBoolean(name: string, fallback: boolean): boolean {
  const raw = process.env[name];
  if (raw === undefined) return fallback;
  return raw.toLowerCase() === "true" || raw === "1";
}

export function loadEnvConfig(): EnvConfig {
  return {
    apiKey: readEnvString("API_KEY", ""),
    maxRetries: readEnvNumber("MAX_RETRIES", 3),
    enableCache: readEnvBoolean("ENABLE_CACHE", true),
    cacheTimeout: readEnvNumber("CACHE_TIMEOUT", 60000),
  };
}

export function validateEnvConfig(config: EnvConfig): void {
  if (!config.apiKey) {
    throw new Error("API_KEY environment variable is required");
  }
  if (config.maxRetries < 0) {
    throw new Error("MAX_RETRIES must be non-negative");
  }
  if (config.cacheTimeout < 0) {
    throw new Error("CACHE_TIMEOUT must be non-negative");
  }
}""",
    },
]

# Path Multiplier Patterns
PATH_MULTIPLIER_PATTERNS = [
    {
        "instruction": "Implement path-based boost multipliers sorted by prefix length",
        "input": """interface PathMultiplier {
  prefix: string;
  multiplier: number;
}""",
        "output": """// Path multipliers sorted by prefix length (longest first for correct matching)
export const PATH_MULTIPLIERS: PathMultiplier[] = [
  { prefix: "src/vs/workbench/contrib/", multiplier: 2.4 },
  { prefix: "src/vs/workbench/", multiplier: 2.2 },
  { prefix: "src/vs/platform/", multiplier: 2.1 },
  { prefix: "src/vs/editor/", multiplier: 2.0 },
  { prefix: "src/vs/", multiplier: 1.8 },
  { prefix: "src/components/", multiplier: 1.3 },
  { prefix: "src/", multiplier: 1.0 },
];

export function getPathMultiplier(filePath: string): number {
  const normalizedPath = filePath.replace(/\\\\/g, "/");

  for (const entry of PATH_MULTIPLIERS) {
    if (normalizedPath.startsWith(entry.prefix)) {
      return entry.multiplier;
    }
  }

  return 1.0; // Default multiplier
}

export function applyPathBoost(score: number, filePath: string): number {
  const multiplier = getPathMultiplier(filePath);
  return score * multiplier;
}""",
    },
    {
        "instruction": "Create a file type classifier with extension-based multipliers",
        "input": """interface FileTypeConfig {
  extensions: string[];
  multiplier: number;
  category: string;
}""",
        "output": """const FILE_TYPE_CONFIGS: FileTypeConfig[] = [
  { extensions: [".ts", ".tsx"], multiplier: 1.5, category: "typescript" },
  { extensions: [".js", ".jsx"], multiplier: 1.3, category: "javascript" },
  { extensions: [".d.ts"], multiplier: 2.0, category: "types" },
  { extensions: [".json"], multiplier: 0.8, category: "config" },
  { extensions: [".md", ".mdx"], multiplier: 0.5, category: "docs" },
  { extensions: [".test.ts", ".spec.ts"], multiplier: 1.2, category: "test" },
];

export function getFileTypeMultiplier(filePath: string): number {
  const lowerPath = filePath.toLowerCase();

  // Check longer extensions first (e.g., .test.ts before .ts)
  const sortedConfigs = [...FILE_TYPE_CONFIGS].sort(
    (a, b) => Math.max(...b.extensions.map((e) => e.length)) - Math.max(...a.extensions.map((e) => e.length))
  );

  for (const config of sortedConfigs) {
    if (config.extensions.some((ext) => lowerPath.endsWith(ext))) {
      return config.multiplier;
    }
  }

  return 1.0;
}

export function getFileCategory(filePath: string): string {
  const lowerPath = filePath.toLowerCase();

  for (const config of FILE_TYPE_CONFIGS) {
    if (config.extensions.some((ext) => lowerPath.endsWith(ext))) {
      return config.category;
    }
  }

  return "unknown";
}""",
    },
    {
        "instruction": "Implement a directory depth-based scoring system",
        "input": """// Score files based on directory depth
// Shallower files should have higher scores""",
        "output": """export function getDirectoryDepth(filePath: string): number {
  const normalizedPath = filePath.replace(/\\\\/g, "/");
  const parts = normalizedPath.split("/").filter((p) => p.length > 0);
  return parts.length - 1; // -1 to not count the filename
}

export function getDepthMultiplier(filePath: string, maxDepth: number = 10): number {
  const depth = getDirectoryDepth(filePath);
  // Linear decay: depth 0 = 1.0, depth maxDepth = 0.5
  const decayFactor = 0.5;
  const multiplier = 1.0 - (depth / maxDepth) * decayFactor;
  return Math.max(0.5, Math.min(1.0, multiplier));
}

export interface ScoredFile {
  path: string;
  baseScore: number;
  depthMultiplier: number;
  finalScore: number;
}

export function scoreFilesByDepth(files: string[], baseScore: number = 1.0): ScoredFile[] {
  return files.map((path) => {
    const depthMultiplier = getDepthMultiplier(path);
    return {
      path,
      baseScore,
      depthMultiplier,
      finalScore: baseScore * depthMultiplier,
    };
  }).sort((a, b) => b.finalScore - a.finalScore);
}""",
    },
]

ALL_KIRI_PATTERNS = (
    SINGLETON_REGISTRY_PATTERNS +
    DISCRIMINATED_UNION_PATTERNS +
    WEIGHTED_SCORING_PATTERNS +
    CONFIG_VALIDATION_PATTERNS +
    PATH_MULTIPLIER_PATTERNS
)


def generate_variations(patterns: list[dict]) -> list[dict]:
    """Generate slight variations of patterns for more training data."""
    variations = []
    for pattern in patterns:
        # Original
        variations.append(pattern)

        # Variation with slightly modified instruction
        if "implement" in pattern["instruction"].lower():
            var = pattern.copy()
            var["instruction"] = pattern["instruction"].replace("Implement", "Create")
            variations.append(var)
        elif "create" in pattern["instruction"].lower():
            var = pattern.copy()
            var["instruction"] = pattern["instruction"].replace("Create", "Implement")
            variations.append(var)

    return variations


def create_alpaca_entry(pattern: dict) -> dict:
    """Convert pattern to text format (same as common-patterns)."""
    text = f"""### Instruction:
{pattern["instruction"]}

### Input:
{pattern["input"].strip()}

### Response:
{pattern["output"].strip()}"""
    return {"text": text}


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "kiri-patterns"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Collecting Kiri-Specific Patterns")
    print("=" * 70)

    # Generate variations
    all_patterns = generate_variations(ALL_KIRI_PATTERNS)
    random.shuffle(all_patterns)

    print(f"Total patterns (with variations): {len(all_patterns)}")

    # Split into train/valid
    split_idx = int(len(all_patterns) * 0.9)
    train_patterns = all_patterns[:split_idx]
    valid_patterns = all_patterns[split_idx:]

    # Convert to Alpaca format
    train_data = [create_alpaca_entry(p) for p in train_patterns]
    valid_data = [create_alpaca_entry(p) for p in valid_patterns]

    # Save
    train_file = output_dir / "train.jsonl"
    valid_file = output_dir / "valid.jsonl"

    with open(train_file, "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(valid_file, "w") as f:
        for entry in valid_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nPattern categories:")
    print(f"  Singleton Registry: {len(SINGLETON_REGISTRY_PATTERNS)}")
    print(f"  Discriminated Union: {len(DISCRIMINATED_UNION_PATTERNS)}")
    print(f"  Weighted Scoring: {len(WEIGHTED_SCORING_PATTERNS)}")
    print(f"  Config Validation: {len(CONFIG_VALIDATION_PATTERNS)}")
    print(f"  Path Multipliers: {len(PATH_MULTIPLIER_PATTERNS)}")

    print(f"\nOutput:")
    print(f"  Train: {train_file} ({len(train_data)} examples)")
    print(f"  Valid: {valid_file} ({len(valid_data)} examples)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
