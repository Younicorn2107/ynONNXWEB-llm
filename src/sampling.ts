/**
 * Choosing the next token: the core's logits processors, then argmax or a seeded draw. One row at a
 * time — a chat generates one sequence.
 */

import {
  MinPWarper,
  PresenceFrequencyPenaltyProcessor,
  RepetitionPenaltyProcessor,
  TemperatureWarper,
  TopKWarper,
  TopPWarper,
  applyProcessors,
  seededRandom,
  type LogitsProcessor,
} from "@youni/onnxweb-core/core";

export interface SamplingSettings {
  /** 0 is greedy. */
  temperature?: number;
  topK?: number;
  topP?: number;
  minP?: number;
  repetitionPenalty?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  /** A seed makes sampling reproducible; null draws from Math.random. */
  seed?: number | null;
}

export class Sampler {
  readonly #processors: LogitsProcessor[];
  readonly #random: () => number;
  readonly greedy: boolean;

  constructor(settings: SamplingSettings = {}) {
    const temperature = settings.temperature ?? 0;
    this.greedy = temperature <= 0;
    this.#processors = [
      new RepetitionPenaltyProcessor(settings.repetitionPenalty ?? 1),
      new PresenceFrequencyPenaltyProcessor(settings.presencePenalty ?? 0, settings.frequencyPenalty ?? 0),
    ];
    if (!this.greedy) {
      this.#processors.push(new TemperatureWarper(temperature), new TopKWarper(settings.topK ?? 0), new TopPWarper(settings.topP ?? 1), new MinPWarper(settings.minP ?? 0));
    }
    this.#random = settings.seed === null || settings.seed === undefined ? Math.random : seededRandom(settings.seed);
  }

  /** The next token after `history` (the whole context: prompt and generated so far). */
  next(logits: Float32Array, history: readonly number[]): number {
    applyProcessors(this.#processors, { data: logits, rows: 1, vocab: logits.length }, [history]);
    if (this.greedy) return argmax(logits);
    let max = Number.NEGATIVE_INFINITY;
    for (const v of logits) if (v > max) max = v;
    let total = 0;
    const weights = new Float64Array(logits.length);
    for (let i = 0; i < logits.length; i += 1) {
      const w = Number.isFinite(logits[i]!) ? Math.exp(logits[i]! - max) : 0;
      weights[i] = w;
      total += w;
    }
    let threshold = this.#random() * total;
    for (let i = 0; i < weights.length; i += 1) {
      threshold -= weights[i]!;
      if (threshold <= 0 && weights[i]! > 0) return i;
    }
    return argmax(logits);
  }
}

export function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i += 1) if (values[i]! > values[best]!) best = i;
  return best;
}
