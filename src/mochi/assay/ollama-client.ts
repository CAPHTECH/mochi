/**
 * Ollama Client for LLM inference via local Ollama server.
 *
 * Used for comparison evaluation: baseline LLM vs mochi-assisted LLM.
 */

export interface OllamaGenerateOptions {
  /** Model name (e.g., "gpt-oss:120b") */
  model: string;
  /** Temperature for generation (default: 0.1 for deterministic output) */
  temperature?: number;
  /** Maximum tokens to generate (default: 512) */
  maxTokens?: number;
  /** System prompt to prepend */
  systemPrompt?: string;
  /** Request timeout in milliseconds (default: 300000 = 5 minutes) */
  timeoutMs?: number;
}

export interface OllamaGenerateResult {
  /** Generated response text */
  response: string;
  /** Total duration in nanoseconds */
  totalDuration: number;
  /** Number of tokens in prompt */
  promptEvalCount: number;
  /** Number of tokens generated */
  evalCount: number;
  /** Duration of generation in nanoseconds */
  evalDuration: number;
}

interface OllamaApiResponse {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
  context?: number[];
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

export class OllamaClient {
  private endpoint: string;
  private defaultModel: string;

  constructor(options: {
    endpoint?: string;
    defaultModel?: string;
  } = {}) {
    this.endpoint = options.endpoint ?? "http://localhost:11434";
    this.defaultModel = options.defaultModel ?? "gpt-oss:120b";
  }

  /**
   * Check if Ollama server is available
   */
  async ping(): Promise<boolean> {
    try {
      const response = await fetch(`${this.endpoint}/api/tags`, {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * List available models
   */
  async listModels(): Promise<string[]> {
    const response = await fetch(`${this.endpoint}/api/tags`, {
      method: "GET",
    });

    if (!response.ok) {
      throw new Error(`Failed to list models: ${response.statusText}`);
    }

    const data = await response.json() as { models: Array<{ name: string }> };
    return data.models.map(m => m.name);
  }

  /**
   * Generate text using Ollama API
   */
  async generate(
    prompt: string,
    options: Partial<OllamaGenerateOptions> = {}
  ): Promise<OllamaGenerateResult> {
    const model = options.model ?? this.defaultModel;
    const temperature = options.temperature ?? 0.1;
    const maxTokens = options.maxTokens ?? 512;
    const timeoutMs = options.timeoutMs ?? 300000; // 5 minutes default

    // Build the full prompt with optional system prompt
    let fullPrompt = prompt;
    if (options.systemPrompt) {
      fullPrompt = `${options.systemPrompt}\n\n${prompt}`;
    }

    const requestBody = {
      model,
      prompt: fullPrompt,
      stream: false,
      options: {
        temperature,
        num_predict: maxTokens,
      },
    };

    const response = await fetch(`${this.endpoint}/api/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(timeoutMs),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Ollama API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json() as OllamaApiResponse;

    return {
      response: data.response,
      totalDuration: data.total_duration ?? 0,
      promptEvalCount: data.prompt_eval_count ?? 0,
      evalCount: data.eval_count ?? 0,
      evalDuration: data.eval_duration ?? 0,
    };
  }

  /**
   * Generate with chat format (for models that support it)
   */
  async chat(
    messages: Array<{ role: "system" | "user" | "assistant"; content: string }>,
    options: Partial<OllamaGenerateOptions> = {}
  ): Promise<OllamaGenerateResult> {
    const model = options.model ?? this.defaultModel;
    const temperature = options.temperature ?? 0.1;
    const maxTokens = options.maxTokens ?? 512;
    const timeoutMs = options.timeoutMs ?? 300000;

    const requestBody = {
      model,
      messages,
      stream: false,
      options: {
        temperature,
        num_predict: maxTokens,
      },
    };

    const response = await fetch(`${this.endpoint}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(timeoutMs),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Ollama chat API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json() as {
      message: { content: string };
      total_duration?: number;
      prompt_eval_count?: number;
      eval_count?: number;
      eval_duration?: number;
    };

    return {
      response: data.message.content,
      totalDuration: data.total_duration ?? 0,
      promptEvalCount: data.prompt_eval_count ?? 0,
      evalCount: data.eval_count ?? 0,
      evalDuration: data.eval_duration ?? 0,
    };
  }

  /**
   * Helper: Generate code completion
   */
  async completeCode(
    instruction: string,
    codePrefix: string,
    context?: string,
    options: Partial<OllamaGenerateOptions> = {}
  ): Promise<OllamaGenerateResult> {
    let prompt = `## Instruction\n${instruction}\n\n## Code to complete\n${codePrefix}`;

    if (context) {
      prompt = `## Context\n${context}\n\n${prompt}`;
    }

    prompt += "\n\n## Completion (code only, no explanation):";

    return this.generate(prompt, {
      ...options,
      temperature: options.temperature ?? 0.1, // Low temperature for code
    });
  }
}
