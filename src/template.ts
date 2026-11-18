/** A model's chat template, rendered within bounds. */

import { Template } from "@huggingface/jinja";
import { ConfigurationError } from "@youni/onnxweb-core/core";

/**
 * Longer than any published chat template (Qwen3's is about 4 KB, the longest tool-calling ones
 * about 20 KB).
 */
export const MAX_TEMPLATE_CHARS = 128 * 1024;
/** The numbers `range()` may produce in one render, all calls together. */
export const MAX_RANGE_ITEMS = 100_000;
/** How long one render may take, checked at each `range()`. */
export const MAX_RENDER_MS = 2000;

/**
 * The name `range` is called by after the rewrite: the interpreter's own `range` cannot be replaced
 * (it is declared first).
 */
const BOUNDED_RANGE = "__ynonnxweb_range";

/** Point every `range(...)` call of the parsed template at the bounded one. */
function boundRangeCalls(node: unknown, depth = 0): void {
  if (node === null || typeof node !== "object" || depth > 10_000) return;
  if (Array.isArray(node)) {
    for (const item of node) boundRangeCalls(item, depth + 1);
    return;
  }
  const record = node as { type?: unknown; callee?: { type?: unknown; value?: unknown } };
  if (record.type === "CallExpression" && record.callee?.type === "Identifier" && record.callee.value === "range") record.callee.value = BOUNDED_RANGE;
  for (const value of Object.values(node)) boundRangeCalls(value, depth + 1);
}

export class ChatTemplate {
  readonly #template: Template;

  constructor(readonly source: string) {
    if (source.length > MAX_TEMPLATE_CHARS) {
      throw new ConfigurationError(`the chat template is ${source.length} characters, more than the ${MAX_TEMPLATE_CHARS} a template is allowed`, {
        details: { length: source.length, max: MAX_TEMPLATE_CHARS },
      });
    }
    this.#template = new Template(source);
    boundRangeCalls(this.#template.parsed);
  }

  render(variables: Record<string, unknown>): string {
    let left = MAX_RANGE_ITEMS;
    const started = Date.now();
    const range = (start: number, stop?: number, step = 1): number[] => {
      if (stop === undefined) [start, stop] = [0, start];
      if (![start, stop, step].every(Number.isFinite) || step === 0) throw new ConfigurationError("the chat template called range() with arguments it cannot have");
      const count = Math.max(0, Math.ceil((stop - start) / step));
      if (count > left) throw new ConfigurationError(`the chat template loops further than the ${MAX_RANGE_ITEMS} steps a render is allowed`);
      if (Date.now() - started > MAX_RENDER_MS) throw new ConfigurationError(`the chat template ran for more than ${MAX_RENDER_MS} ms`);
      left -= count;
      return Array.from({ length: count }, (_, i) => start + i * step);
    };
    return this.#template.render({ ...variables, [BOUNDED_RANGE]: range });
  }
}
