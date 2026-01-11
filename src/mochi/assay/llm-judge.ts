/**
 * LLM-as-Judge evaluator for code completion quality.
 *
 * Uses a large LLM (gpt-oss 120B) to evaluate generated code
 * for correctness, convention adherence, and hallucination detection.
 */

import { OllamaClient } from "./ollama-client.js";

export interface JudgeResult {
  /** Overall score (0.0-1.0) */
  score: number;
  /** Detailed reasoning */
  reasoning: string;
  /** Individual criteria scores */
  criteria: {
    apiCorrectness: number;
    conventionAdherence: number;
    syntaxCorrectness: number;
    semanticAppropriateness: number;
  };
  /** Detected issues */
  issues: string[];
}

export interface JudgeInput {
  instruction: string;
  input: string;
  context?: string;
  output: string;
  expectedKeywords?: string[];
  forbiddenKeywords?: string[];
}

export class LLMJudge {
  private ollamaClient: OllamaClient;
  private model: string;

  constructor(options: {
    ollamaEndpoint?: string;
    model?: string;
  } = {}) {
    this.ollamaClient = new OllamaClient({
      endpoint: options.ollamaEndpoint ?? "http://localhost:11434",
    });
    this.model = options.model ?? "gpt-oss:120b";
  }

  /**
   * Evaluate code completion quality using LLM
   */
  async evaluate(input: JudgeInput): Promise<JudgeResult> {
    const prompt = this.buildJudgePrompt(input);

    const result = await this.ollamaClient.generate(prompt, {
      model: this.model,
      temperature: 0.0, // Deterministic for consistent evaluation
      maxTokens: 512,
    });

    return this.parseJudgeResponse(result.response);
  }

  /**
   * Build the judge prompt
   */
  private buildJudgePrompt(input: JudgeInput): string {
    let prompt = `You are a code quality evaluator. Evaluate the generated code completion.

## Task
${input.instruction}

## Code Input
${input.input}
`;

    if (input.context) {
      prompt += `
## Domain Context (ground truth)
${input.context}
`;
    }

    prompt += `
## Generated Output
${input.output}
`;

    if (input.expectedKeywords && input.expectedKeywords.length > 0) {
      prompt += `
## Expected Keywords (should appear)
${input.expectedKeywords.join(", ")}
`;
    }

    if (input.forbiddenKeywords && input.forbiddenKeywords.length > 0) {
      prompt += `
## Forbidden Keywords (should NOT appear - hallucination indicators)
${input.forbiddenKeywords.join(", ")}
`;
    }

    prompt += `
## Evaluation Criteria
Rate each from 0.0 to 1.0:

1. API_CORRECTNESS: Uses methods/APIs listed in context, not hallucinated ones
2. CONVENTION_ADHERENCE: Follows project conventions from context
3. SYNTAX_CORRECTNESS: Syntactically valid code
4. SEMANTIC_APPROPRIATENESS: Code makes sense for the task

## Response Format (REQUIRED - follow exactly)
API_CORRECTNESS: X.X
CONVENTION_ADHERENCE: X.X
SYNTAX_CORRECTNESS: X.X
SEMANTIC_APPROPRIATENESS: X.X
OVERALL: X.X
ISSUES: [comma-separated list or "none"]
REASONING: Brief explanation

Evaluate now:`;

    return prompt;
  }

  /**
   * Parse the judge response
   */
  private parseJudgeResponse(response: string): JudgeResult {
    const lines = response.split("\n");
    const result: JudgeResult = {
      score: 0,
      reasoning: "",
      criteria: {
        apiCorrectness: 0,
        conventionAdherence: 0,
        syntaxCorrectness: 0,
        semanticAppropriateness: 0,
      },
      issues: [],
    };

    for (const line of lines) {
      const trimmed = line.trim();

      if (trimmed.startsWith("API_CORRECTNESS:")) {
        result.criteria.apiCorrectness = this.parseScore(trimmed);
      } else if (trimmed.startsWith("CONVENTION_ADHERENCE:")) {
        result.criteria.conventionAdherence = this.parseScore(trimmed);
      } else if (trimmed.startsWith("SYNTAX_CORRECTNESS:")) {
        result.criteria.syntaxCorrectness = this.parseScore(trimmed);
      } else if (trimmed.startsWith("SEMANTIC_APPROPRIATENESS:")) {
        result.criteria.semanticAppropriateness = this.parseScore(trimmed);
      } else if (trimmed.startsWith("OVERALL:")) {
        result.score = this.parseScore(trimmed);
      } else if (trimmed.startsWith("ISSUES:")) {
        const issuesStr = trimmed.substring("ISSUES:".length).trim();
        if (issuesStr.toLowerCase() !== "none" && issuesStr !== "[]") {
          result.issues = issuesStr
            .replace(/^\[|\]$/g, "")
            .split(",")
            .map(s => s.trim())
            .filter(s => s.length > 0);
        }
      } else if (trimmed.startsWith("REASONING:")) {
        result.reasoning = trimmed.substring("REASONING:".length).trim();
      }
    }

    // If OVERALL wasn't parsed, calculate from criteria
    if (result.score === 0) {
      const { apiCorrectness, conventionAdherence, syntaxCorrectness, semanticAppropriateness } =
        result.criteria;
      result.score =
        (apiCorrectness + conventionAdherence + syntaxCorrectness + semanticAppropriateness) / 4;
    }

    return result;
  }

  /**
   * Parse a score from a line like "CRITERIA: 0.8"
   */
  private parseScore(line: string): number {
    const match = line.match(/:\s*([\d.]+)/);
    if (match) {
      const score = parseFloat(match[1]);
      return Math.max(0, Math.min(1, score));
    }
    return 0;
  }
}

/**
 * Simple AST validator for TypeScript/JavaScript code
 * Note: This is a basic check, not a full AST parse
 */
export function validateSyntax(code: string, language: string = "typescript"): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  if (language === "typescript" || language === "javascript") {
    // Check for balanced braces
    let braceCount = 0;
    let parenCount = 0;
    let bracketCount = 0;

    for (const char of code) {
      switch (char) {
        case "{":
          braceCount++;
          break;
        case "}":
          braceCount--;
          break;
        case "(":
          parenCount++;
          break;
        case ")":
          parenCount--;
          break;
        case "[":
          bracketCount++;
          break;
        case "]":
          bracketCount--;
          break;
      }

      // Check for negative counts (closing before opening)
      if (braceCount < 0) errors.push("Unexpected }");
      if (parenCount < 0) errors.push("Unexpected )");
      if (bracketCount < 0) errors.push("Unexpected ]");
    }

    if (braceCount > 0) errors.push(`Missing ${braceCount} closing brace(s)`);
    if (parenCount > 0) errors.push(`Missing ${parenCount} closing paren(s)`);
    if (bracketCount > 0) errors.push(`Missing ${bracketCount} closing bracket(s)`);

    // Check for common syntax errors
    if (/\b(const|let|var)\s+\d/.test(code)) {
      errors.push("Variable name cannot start with digit");
    }

    if (/=\s*=\s*=\s*=/.test(code)) {
      errors.push("Invalid comparison operator");
    }

    // Check for incomplete statements
    if (/\bawait\s*$/.test(code.trim())) {
      errors.push("Incomplete await statement");
    }

    if (/\breturn\s*$/.test(code.trim()) && !code.trim().endsWith(";")) {
      // This might be intentional (return;), so don't flag as error
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Combined evaluation using keyword matching, AST, and LLM judge
 */
export interface CombinedEvaluationResult {
  keywordScore: number;
  syntaxValid: boolean;
  syntaxErrors: string[];
  judgeScore: number;
  judgeReasoning: string;
  overallScore: number;
}

export async function combinedEvaluate(
  input: JudgeInput,
  judge: LLMJudge,
  language: string = "typescript"
): Promise<CombinedEvaluationResult> {
  // 1. Keyword-based evaluation
  const expected = input.expectedKeywords ?? [];
  const forbidden = input.forbiddenKeywords ?? [];
  const outputLower = input.output.toLowerCase();

  const foundKeywords = expected.filter(k => outputLower.includes(k.toLowerCase()));
  const violations = forbidden.filter(k => outputLower.includes(k.toLowerCase()));

  let keywordScore = expected.length > 0 ? foundKeywords.length / expected.length : 1.0;
  keywordScore *= Math.max(0, 1 - violations.length * 0.2);

  // 2. Syntax validation
  const syntaxResult = validateSyntax(input.output, language);

  // 3. LLM Judge evaluation
  const judgeResult = await judge.evaluate(input);

  // 4. Calculate overall score (weighted average)
  const weights = {
    keyword: 0.3,
    syntax: 0.2,
    judge: 0.5,
  };

  const overallScore =
    weights.keyword * keywordScore +
    weights.syntax * (syntaxResult.valid ? 1.0 : 0.5) +
    weights.judge * judgeResult.score;

  return {
    keywordScore,
    syntaxValid: syntaxResult.valid,
    syntaxErrors: syntaxResult.errors,
    judgeScore: judgeResult.score,
    judgeReasoning: judgeResult.reasoning,
    overallScore: Math.max(0, Math.min(1, overallScore)),
  };
}
