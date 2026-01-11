#!/usr/bin/env python3
"""Collect common TypeScript/JavaScript patterns for Base Adapter training.

This script collects learnable patterns from:
1. Public GitHub repositories (TypeScript/JavaScript)
2. Curated pattern examples

Patterns are categorized into:
- error-handling: try-catch, Result types, typed errors
- async-await: Promise handling, async functions
- type-safety: type annotations, type guards
- null-safety: optional chaining, nullish coalescing
- validation: zod, assertions, input validation
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class PatternExample:
    """A single pattern example for training."""

    instruction: str
    input: str  # before code or context
    output: str  # after code or completion
    pattern_type: str
    source: str = "synthetic"


# Pattern templates for each category
ERROR_HANDLING_PATTERNS = [
    {
        "instruction": "Add error handling to this function",
        "input": """async function fetchUser(id: string) {
  const response = await fetch(`/api/users/${id}`);
  const data = await response.json();
  return data;
}""",
        "output": """async function fetchUser(id: string): Promise<User> {
  try {
    const response = await fetch(`/api/users/${id}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch user: ${response.status}`);
    }
    const data = await response.json();
    return data as User;
  } catch (error) {
    if (error instanceof Error) {
      throw new FetchError(`User fetch failed: ${error.message}`, { cause: error });
    }
    throw new FetchError('Unknown error fetching user');
  }
}""",
    },
    {
        "instruction": "Wrap this database operation with proper error handling",
        "input": """async function saveRecord(data: Record) {
  const result = await db.insert(data);
  return result.id;
}""",
        "output": """async function saveRecord(data: Record): Promise<string> {
  try {
    const result = await db.insert(data);
    if (!result.id) {
      throw new DatabaseError('Insert succeeded but no ID returned');
    }
    return result.id;
  } catch (error) {
    if (error instanceof DatabaseError) {
      throw error;
    }
    throw new DatabaseError('Failed to save record', { cause: error });
  }
}""",
    },
    {
        "instruction": "Add try-catch with typed error handling",
        "input": """function parseConfig(raw: string) {
  const config = JSON.parse(raw);
  return config;
}""",
        "output": """function parseConfig(raw: string): Config {
  try {
    const config = JSON.parse(raw);
    return config as Config;
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new ConfigError(`Invalid JSON: ${error.message}`);
    }
    throw new ConfigError('Failed to parse config', { cause: error });
  }
}""",
    },
    {
        "instruction": "Add error recovery to this file operation",
        "input": """async function readFile(path: string) {
  const content = await fs.readFile(path, 'utf-8');
  return content;
}""",
        "output": """async function readFile(path: string): Promise<string> {
  try {
    const content = await fs.readFile(path, 'utf-8');
    return content;
  } catch (error) {
    if (error instanceof Error && 'code' in error) {
      const fsError = error as NodeJS.ErrnoException;
      if (fsError.code === 'ENOENT') {
        throw new FileNotFoundError(`File not found: ${path}`);
      }
      if (fsError.code === 'EACCES') {
        throw new PermissionError(`Permission denied: ${path}`);
      }
    }
    throw new FileError(`Failed to read file: ${path}`, { cause: error });
  }
}""",
    },
    {
        "instruction": "Handle potential errors in this API call",
        "input": """async function createOrder(items: Item[]) {
  const order = await api.post('/orders', { items });
  await sendConfirmationEmail(order.email);
  return order;
}""",
        "output": """async function createOrder(items: Item[]): Promise<Order> {
  let order: Order;
  try {
    order = await api.post('/orders', { items });
  } catch (error) {
    throw new OrderError('Failed to create order', { cause: error });
  }

  try {
    await sendConfirmationEmail(order.email);
  } catch (emailError) {
    // Log but don't fail - order was created successfully
    console.error('Failed to send confirmation email:', emailError);
  }

  return order;
}""",
    },
]

ASYNC_AWAIT_PATTERNS = [
    {
        "instruction": "Convert this callback-based code to async/await",
        "input": """function loadData(callback) {
  fetch('/api/data')
    .then(response => response.json())
    .then(data => callback(null, data))
    .catch(error => callback(error, null));
}""",
        "output": """async function loadData(): Promise<Data> {
  const response = await fetch('/api/data');
  const data = await response.json();
  return data;
}""",
    },
    {
        "instruction": "Refactor to use async/await pattern",
        "input": """function processItems(items) {
  return Promise.all(items.map(item => {
    return fetch(`/api/process/${item.id}`)
      .then(res => res.json());
  }));
}""",
        "output": """async function processItems(items: Item[]): Promise<ProcessedItem[]> {
  const results = await Promise.all(
    items.map(async (item) => {
      const response = await fetch(`/api/process/${item.id}`);
      return response.json();
    })
  );
  return results;
}""",
    },
    {
        "instruction": "Convert Promise chains to async/await",
        "input": """function getUserPosts(userId) {
  return getUser(userId)
    .then(user => getPosts(user.id))
    .then(posts => posts.filter(p => p.published))
    .then(posts => posts.map(p => ({ ...p, author: userId })));
}""",
        "output": """async function getUserPosts(userId: string): Promise<Post[]> {
  const user = await getUser(userId);
  const posts = await getPosts(user.id);
  const publishedPosts = posts.filter(p => p.published);
  return publishedPosts.map(p => ({ ...p, author: userId }));
}""",
    },
    {
        "instruction": "Modernize this async code using await",
        "input": """function fetchWithRetry(url, retries = 3) {
  return fetch(url).catch(error => {
    if (retries > 0) {
      return fetchWithRetry(url, retries - 1);
    }
    throw error;
  });
}""",
        "output": """async function fetchWithRetry(url: string, retries = 3): Promise<Response> {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const response = await fetch(url);
      return response;
    } catch (error) {
      if (attempt === retries) {
        throw error;
      }
      await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
    }
  }
  throw new Error('Unreachable');
}""",
    },
    {
        "instruction": "Convert to async/await with proper error handling",
        "input": """function saveAndNotify(data) {
  return db.save(data)
    .then(result => {
      return notify(result.id).then(() => result);
    })
    .catch(err => {
      console.error(err);
      throw err;
    });
}""",
        "output": """async function saveAndNotify(data: Data): Promise<Result> {
  try {
    const result = await db.save(data);
    await notify(result.id);
    return result;
  } catch (error) {
    console.error('Save and notify failed:', error);
    throw error;
  }
}""",
    },
]

TYPE_SAFETY_PATTERNS = [
    {
        "instruction": "Add type annotations to this code",
        "input": """function processUser(user) {
  const fullName = user.firstName + ' ' + user.lastName;
  const age = new Date().getFullYear() - user.birthYear;
  return { fullName, age, email: user.email };
}""",
        "output": """interface User {
  firstName: string;
  lastName: string;
  birthYear: number;
  email: string;
}

interface ProcessedUser {
  fullName: string;
  age: number;
  email: string;
}

function processUser(user: User): ProcessedUser {
  const fullName = user.firstName + ' ' + user.lastName;
  const age = new Date().getFullYear() - user.birthYear;
  return { fullName, age, email: user.email };
}""",
    },
    {
        "instruction": "Add type guards where appropriate",
        "input": """function handleResponse(response) {
  if (response.error) {
    throw new Error(response.error.message);
  }
  return response.data;
}""",
        "output": """interface SuccessResponse<T> {
  data: T;
  error?: never;
}

interface ErrorResponse {
  data?: never;
  error: { message: string; code: string };
}

type ApiResponse<T> = SuccessResponse<T> | ErrorResponse;

function isErrorResponse<T>(response: ApiResponse<T>): response is ErrorResponse {
  return 'error' in response && response.error !== undefined;
}

function handleResponse<T>(response: ApiResponse<T>): T {
  if (isErrorResponse(response)) {
    throw new Error(response.error.message);
  }
  return response.data;
}""",
    },
    {
        "instruction": "Strengthen the types in this code",
        "input": """function formatDate(date, format) {
  const d = new Date(date);
  if (format === 'short') {
    return d.toLocaleDateString();
  }
  return d.toISOString();
}""",
        "output": """type DateFormat = 'short' | 'iso' | 'long';

function formatDate(date: Date | string | number, format: DateFormat): string {
  const d = date instanceof Date ? date : new Date(date);

  switch (format) {
    case 'short':
      return d.toLocaleDateString();
    case 'long':
      return d.toLocaleDateString(undefined, {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric'
      });
    case 'iso':
      return d.toISOString();
  }
}""",
    },
    {
        "instruction": "Improve type safety of this function",
        "input": """function getProperty(obj, key) {
  return obj[key];
}""",
        "output": """function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}""",
    },
    {
        "instruction": "Add proper TypeScript types",
        "input": """const handlers = {
  click: (e) => console.log('clicked', e.target),
  hover: (e) => console.log('hovered', e.target),
  focus: (e) => console.log('focused', e.target),
};

function dispatch(event, handler) {
  handlers[handler](event);
}""",
        "output": """type EventHandler = (event: Event) => void;
type HandlerName = 'click' | 'hover' | 'focus';

const handlers: Record<HandlerName, EventHandler> = {
  click: (e: Event) => console.log('clicked', e.target),
  hover: (e: Event) => console.log('hovered', e.target),
  focus: (e: Event) => console.log('focused', e.target),
};

function dispatch(event: Event, handler: HandlerName): void {
  handlers[handler](event);
}""",
    },
]

NULL_SAFETY_PATTERNS = [
    {
        "instruction": "Add null safety checks to this code",
        "input": """function getUserEmail(user) {
  return user.profile.contact.email;
}""",
        "output": """function getUserEmail(user: User | null | undefined): string | undefined {
  return user?.profile?.contact?.email;
}""",
    },
    {
        "instruction": "Make this code null-safe using optional chaining",
        "input": """function getDisplayName(user) {
  if (user && user.profile && user.profile.displayName) {
    return user.profile.displayName;
  }
  return user && user.username ? user.username : 'Anonymous';
}""",
        "output": """function getDisplayName(user: User | null | undefined): string {
  return user?.profile?.displayName ?? user?.username ?? 'Anonymous';
}""",
    },
    {
        "instruction": "Add proper null checks to prevent runtime errors",
        "input": """function processItems(items) {
  return items.map(item => item.value * 2);
}""",
        "output": """function processItems(items: Item[] | null | undefined): number[] {
  if (!items?.length) {
    return [];
  }
  return items.map(item => (item?.value ?? 0) * 2);
}""",
    },
    {
        "instruction": "Handle null and undefined values safely",
        "input": """function getConfig(options) {
  const timeout = options.timeout;
  const retries = options.retries;
  const baseUrl = options.baseUrl;
  return { timeout, retries, baseUrl };
}""",
        "output": """interface ConfigOptions {
  timeout?: number;
  retries?: number;
  baseUrl?: string;
}

interface Config {
  timeout: number;
  retries: number;
  baseUrl: string;
}

function getConfig(options?: ConfigOptions): Config {
  return {
    timeout: options?.timeout ?? 5000,
    retries: options?.retries ?? 3,
    baseUrl: options?.baseUrl ?? 'https://api.example.com',
  };
}""",
    },
    {
        "instruction": "Add defensive null handling",
        "input": """function findUser(users, id) {
  const user = users.find(u => u.id === id);
  return user.name;
}""",
        "output": """function findUser(users: User[], id: string): string | undefined {
  const user = users.find(u => u.id === id);
  return user?.name;
}

// Or with explicit handling:
function findUserOrThrow(users: User[], id: string): string {
  const user = users.find(u => u.id === id);
  if (!user) {
    throw new Error(`User not found: ${id}`);
  }
  return user.name;
}""",
    },
]

VALIDATION_PATTERNS = [
    {
        "instruction": "Add input validation to this function",
        "input": """function createUser(data) {
  return db.users.create({
    name: data.name,
    email: data.email,
    age: data.age,
  });
}""",
        "output": """import { z } from 'zod';

const CreateUserSchema = z.object({
  name: z.string().min(1).max(100),
  email: z.string().email(),
  age: z.number().int().min(0).max(150),
});

type CreateUserInput = z.infer<typeof CreateUserSchema>;

function createUser(data: unknown): Promise<User> {
  const validated = CreateUserSchema.parse(data);
  return db.users.create({
    name: validated.name,
    email: validated.email,
    age: validated.age,
  });
}""",
    },
    {
        "instruction": "Add runtime validation using zod",
        "input": """async function handleRequest(body) {
  const { userId, action, payload } = body;
  return processAction(userId, action, payload);
}""",
        "output": """import { z } from 'zod';

const RequestSchema = z.object({
  userId: z.string().uuid(),
  action: z.enum(['create', 'update', 'delete']),
  payload: z.record(z.unknown()),
});

async function handleRequest(body: unknown): Promise<Result> {
  const result = RequestSchema.safeParse(body);
  if (!result.success) {
    throw new ValidationError('Invalid request', result.error.issues);
  }
  const { userId, action, payload } = result.data;
  return processAction(userId, action, payload);
}""",
    },
    {
        "instruction": "Validate the input parameters",
        "input": """function calculateDiscount(price, percentage) {
  return price * (percentage / 100);
}""",
        "output": """function calculateDiscount(price: number, percentage: number): number {
  if (typeof price !== 'number' || isNaN(price)) {
    throw new TypeError('Price must be a valid number');
  }
  if (typeof percentage !== 'number' || isNaN(percentage)) {
    throw new TypeError('Percentage must be a valid number');
  }
  if (price < 0) {
    throw new RangeError('Price cannot be negative');
  }
  if (percentage < 0 || percentage > 100) {
    throw new RangeError('Percentage must be between 0 and 100');
  }
  return price * (percentage / 100);
}""",
    },
    {
        "instruction": "Add schema validation to this code",
        "input": """function updateSettings(settings) {
  localStorage.setItem('settings', JSON.stringify(settings));
}""",
        "output": """import { z } from 'zod';

const SettingsSchema = z.object({
  theme: z.enum(['light', 'dark', 'system']).default('system'),
  language: z.string().length(2).default('en'),
  notifications: z.boolean().default(true),
  fontSize: z.number().min(10).max(24).default(14),
});

type Settings = z.infer<typeof SettingsSchema>;

function updateSettings(settings: unknown): Settings {
  const validated = SettingsSchema.parse(settings);
  localStorage.setItem('settings', JSON.stringify(validated));
  return validated;
}""",
    },
    {
        "instruction": "Add assertions to verify input",
        "input": """function divide(a, b) {
  return a / b;
}""",
        "output": """function assertNumber(value: unknown, name: string): asserts value is number {
  if (typeof value !== 'number' || isNaN(value)) {
    throw new TypeError(`${name} must be a valid number`);
  }
}

function assertNonZero(value: number, name: string): void {
  if (value === 0) {
    throw new RangeError(`${name} cannot be zero`);
  }
}

function divide(a: unknown, b: unknown): number {
  assertNumber(a, 'Dividend');
  assertNumber(b, 'Divisor');
  assertNonZero(b, 'Divisor');
  return a / b;
}""",
    },
]


def generate_variations(patterns: list[dict], pattern_type: str, count: int) -> list[PatternExample]:
    """Generate variations of pattern examples."""
    examples = []

    # First, add all base patterns
    for pattern in patterns:
        examples.append(PatternExample(
            instruction=pattern["instruction"],
            input=pattern["input"],
            output=pattern["output"],
            pattern_type=pattern_type,
            source="curated",
        ))

    # Generate additional variations by shuffling instruction templates
    instruction_templates = {
        "error-handling": [
            "Add error handling to this function",
            "Wrap this code with try-catch",
            "Add proper error handling",
            "Handle potential errors in this code",
            "Add error recovery",
            "Implement error handling following best practices",
        ],
        "async-await": [
            "Convert to async/await",
            "Refactor using async/await pattern",
            "Modernize this async code",
            "Convert Promise chains to async/await",
            "Use async/await instead of callbacks",
        ],
        "type-safety": [
            "Add type annotations",
            "Improve type safety",
            "Add proper TypeScript types",
            "Add type guards",
            "Strengthen the types",
            "Make this code type-safe",
        ],
        "null-safety": [
            "Add null safety checks",
            "Make this code null-safe",
            "Add proper null handling",
            "Handle null and undefined safely",
            "Add defensive null checks",
            "Use optional chaining",
        ],
        "validation": [
            "Add input validation",
            "Add runtime validation",
            "Validate the input parameters",
            "Add schema validation",
            "Add assertions to verify input",
            "Validate data before processing",
        ],
    }

    templates = instruction_templates.get(pattern_type, [])

    while len(examples) < count:
        base = random.choice(patterns)
        instruction = random.choice(templates) if templates else base["instruction"]

        examples.append(PatternExample(
            instruction=instruction,
            input=base["input"],
            output=base["output"],
            pattern_type=pattern_type,
            source="variation",
        ))

    return examples[:count]


def save_patterns(examples: list[PatternExample], output_dir: Path, pattern_type: str) -> None:
    """Save patterns to JSONL file in mlx-lm compatible format."""
    output_file = output_dir / pattern_type / "train.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for example in examples:
            # mlx-lm format: single "text" field
            text = (
                f"### Instruction:\n{example.instruction}\n\n"
                f"### Input:\n{example.input}\n\n"
                f"### Response:\n{example.output}"
            )
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"Saved {len(examples)} {pattern_type} patterns to {output_file}")


def main():
    """Main entry point."""
    output_dir = Path(__file__).parent.parent / "data" / "common-patterns"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pattern targets (can be adjusted)
    targets = {
        "error-handling": 100,  # Start smaller for MVP
        "async-await": 100,
        "type-safety": 100,
        "null-safety": 100,
        "validation": 60,
    }

    all_examples = []

    # Generate error-handling patterns
    print("Generating error-handling patterns...")
    error_examples = generate_variations(ERROR_HANDLING_PATTERNS, "error-handling", targets["error-handling"])
    save_patterns(error_examples, output_dir, "error-handling")
    all_examples.extend(error_examples)

    # Generate async-await patterns
    print("Generating async-await patterns...")
    async_examples = generate_variations(ASYNC_AWAIT_PATTERNS, "async-await", targets["async-await"])
    save_patterns(async_examples, output_dir, "async-await")
    all_examples.extend(async_examples)

    # Generate type-safety patterns
    print("Generating type-safety patterns...")
    type_examples = generate_variations(TYPE_SAFETY_PATTERNS, "type-safety", targets["type-safety"])
    save_patterns(type_examples, output_dir, "type-safety")
    all_examples.extend(type_examples)

    # Generate null-safety patterns
    print("Generating null-safety patterns...")
    null_examples = generate_variations(NULL_SAFETY_PATTERNS, "null-safety", targets["null-safety"])
    save_patterns(null_examples, output_dir, "null-safety")
    all_examples.extend(null_examples)

    # Generate validation patterns
    print("Generating validation patterns...")
    validation_examples = generate_variations(VALIDATION_PATTERNS, "validation", targets["validation"])
    save_patterns(validation_examples, output_dir, "validation")
    all_examples.extend(validation_examples)

    # Create combined train.jsonl for Base Adapter training
    combined_file = output_dir / "train.jsonl"
    random.shuffle(all_examples)

    with open(combined_file, "w", encoding="utf-8") as f:
        for example in all_examples:
            text = (
                f"### Instruction:\n{example.instruction}\n\n"
                f"### Input:\n{example.input}\n\n"
                f"### Response:\n{example.output}"
            )
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_examples)} patterns")
    print(f"Combined training file: {combined_file}")

    # Create eval split (10%)
    eval_count = len(all_examples) // 10
    eval_examples = all_examples[:eval_count]
    train_examples = all_examples[eval_count:]

    # Rewrite train.jsonl with proper split
    with open(combined_file, "w", encoding="utf-8") as f:
        for example in train_examples:
            text = (
                f"### Instruction:\n{example.instruction}\n\n"
                f"### Input:\n{example.input}\n\n"
                f"### Response:\n{example.output}"
            )
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    eval_file = output_dir / "valid.jsonl"
    with open(eval_file, "w", encoding="utf-8") as f:
        for example in eval_examples:
            text = (
                f"### Instruction:\n{example.instruction}\n\n"
                f"### Input:\n{example.input}\n\n"
                f"### Response:\n{example.output}"
            )
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"Train: {len(train_examples)} examples -> {combined_file}")
    print(f"Valid: {len(eval_examples)} examples -> {eval_file}")


if __name__ == "__main__":
    main()
