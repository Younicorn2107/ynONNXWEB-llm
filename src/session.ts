/** One decoder-only model and its KV cache, kept between calls. */

import type { ModelBundle, ModelSession } from "@youni/onnxweb-core";
import { CancelledError, GenerationError, yieldToEventLoop, type TensorFactory, type TensorLike } from "@youni/onnxweb-core/core";

const PAST = "past_key_values.";
const PRESENT = "present.";

/**
 * The cache input an output feeds next step: `present.X` → `past_key_values.X`, `present_conv.X` →
 * `past_conv.X`.
 */
function pastOf(output: string): string | null {
  if (output.startsWith(PRESENT)) return PAST + output.slice(PRESENT.length);
  if (output.startsWith("present_")) return `past_${output.slice("present_".length)}`;
  return null;
}

const isState = (name: string) => name.startsWith("past_") && !name.startsWith(PAST);

export type KvData = Float32Array | Uint16Array;

/** The ids this cache holds, and how to get logits for what comes next. */
export class DecoderSession {
  readonly session: ModelSession;
  readonly #past = new Map<string, TensorLike>();
  readonly #pairs = new Map<string, string>();
  #tokens: number[] = [];
  /** Linear-attention states: the cache cannot be cut back, only restarted. */
  readonly recurrent: boolean;

  constructor(
    readonly bundle: ModelBundle,
    readonly Tensor: TensorFactory,
    /** Tokens per prefill run: bounds the activation memory of a long prompt. */
    readonly chunkSize = 512,
    /** A hybrid model's token-embedding graph (and its bundle, closed with this one). */
    readonly embed: { bundle: ModelBundle; session: ModelSession } | null = null,
  ) {
    this.session = bundle.only;
    for (const spec of this.session.outputs) {
      const past = pastOf(spec.name);
      if (past && this.session.inputs.some((s) => s.name === past)) this.#pairs.set(spec.name, past);
    }
    this.recurrent = this.session.inputs.some((s) => isState(s.name));
    if (this.session.inputs.some((s) => s.name === "inputs_embeds") && !embed) {
      throw new GenerationError(`${this.session.filename} takes inputs_embeds: it needs its token-embedding graph`);
    }
    if (this.#pairs.size === 0) {
      throw new GenerationError(`${this.session.filename} has no past_key_values.* inputs paired with present.* outputs: it cannot be stepped`);
    }
  }

  /** The tokens the cache holds. */
  get tokens(): readonly number[] {
    return this.#tokens;
  }

  get length(): number {
    return this.#tokens.length;
  }

  /**
   * Make the cache hold `prompt` and return the logits for the token after it. Reuses the longest
   * common prefix; at least the last token is always run again, so there are logits to return.
   */
  async prime(prompt: readonly number[], signal?: AbortSignal): Promise<{ logits: Float32Array; reused: number }> {
    if (prompt.length === 0) throw new GenerationError("an empty prompt has no next token");
    let common = 0;
    while (common < this.#tokens.length && common < prompt.length && this.#tokens[common] === prompt[common]) common += 1;
    await this.truncate(Math.min(common, prompt.length - 1));

    const keep = this.#tokens.length;
    const logits = await this.feed(prompt.slice(keep), signal);
    return { logits, reused: keep };
  }

  /** Append tokens to the cache; the logits of the last one. */
  async feed(ids: readonly number[], signal?: AbortSignal): Promise<Float32Array> {
    if (ids.length === 0) throw new GenerationError("nothing to feed");
    let logits: Float32Array | null = null;
    for (let start = 0; start < ids.length; start += this.chunkSize) {
      if (signal?.aborted) throw new CancelledError("generation was cancelled");
      const chunk = ids.slice(start, start + this.chunkSize);
      logits = await this.#run(chunk);
      await yieldToEventLoop();
    }
    return logits!;
  }

  async #run(ids: readonly number[]): Promise<Float32Array> {
    const { Tensor } = this;
    const past = this.#tokens.length;
    const length = ids.length;
    const names = new Set(this.session.inputs.map((s) => s.name));
    const feeds: Record<string, unknown> = {};
    const idTensor = new Tensor("int64", BigInt64Array.from(ids, (i) => BigInt(i)), [1, length]);
    let embedded: TensorLike | null = null;
    if (names.has("inputs_embeds")) {
      const out = (await this.embed!.session.run({ [this.embed!.session.inputs[0]!.name]: idTensor })) as Record<string, TensorLike>;
      embedded = out["inputs_embeds"] ?? Object.values(out)[0]!;
      feeds["inputs_embeds"] = embedded;
    } else {
      feeds["input_ids"] = idTensor;
    }
    if (names.has("attention_mask")) feeds["attention_mask"] = new Tensor("int64", new BigInt64Array(past + length).fill(1n), [1, past + length]);
    if (names.has("position_ids")) {
      const axes = this.session.inputs.find((s) => s.name === "position_ids")?.dims?.length === 3 ? 3 : 1;
      const positions = BigInt64Array.from({ length: axes * length }, (_, i) => BigInt(past + (i % length)));
      feeds["position_ids"] = new Tensor("int64", positions, axes === 3 ? [3, 1, length] : [1, length]);
    }
    for (const spec of this.session.inputs) {
      if (!spec.name.startsWith("past_")) continue;
      const cached = this.#past.get(spec.name);
      if (cached) {
        feeds[spec.name] = cached;
        continue;
      }
      if (isState(spec.name)) {
        const dims = (spec.dims ?? []).map((d, axis) => (typeof d === "number" && d > 0 ? d : axis === 0 ? 1 : -1));
        if (dims.includes(-1)) throw new GenerationError(`cannot size an empty state for ${spec.name} (dims ${JSON.stringify(spec.dims)})`);
        const size = dims.reduce((a, b) => a * b, 1);
        const half = spec.type === "float16";
        feeds[spec.name] = new Tensor(half ? "float16" : "float32", half ? new Uint16Array(size) : new Float32Array(size), dims);
        continue;
      }

      const dims = (spec.dims ?? []).map((d, axis) => (typeof d === "number" && d >= 0 ? d : axis === 0 ? 1 : axis === 2 ? 0 : -1));
      if (dims.length !== 4 || dims.includes(-1)) throw new GenerationError(`cannot size an empty cache for ${spec.name} (dims ${JSON.stringify(spec.dims)})`);
      const half = spec.type === "float16";
      feeds[spec.name] = new Tensor(half ? "float16" : "float32", half ? new Uint16Array(0) : new Float32Array(0), dims);
    }
    const missing = [...names].filter((n) => !(n in feeds));
    if (missing.length > 0) throw new GenerationError(`the graph requires ${missing.join(", ")}, which nothing supplies`);
    const produced = (await this.session.run(feeds)) as Record<string, TensorLike>;
    for (const [present, pastName] of this.#pairs) {
      const value = produced[present];
      if (!value) throw new GenerationError(`the graph did not return ${present}`);
      this.#past.get(pastName)?.dispose?.();
      this.#past.set(pastName, value);
    }
    this.#tokens.push(...ids);
    const logitsTensor = produced["logits"] ?? produced[this.session.outputs.find((s) => !s.name.startsWith(PRESENT))?.name ?? ""];
    if (!logitsTensor) throw new GenerationError("the graph returned no logits");
    for (const [name, value] of Object.entries(produced)) if (name !== "logits" && !this.#pairs.has(name) && value !== logitsTensor) value.dispose?.();
    embedded?.dispose?.();
    const logits = await lastRow(logitsTensor);
    logitsTensor.dispose?.();
    return logits;
  }

  /** Keep only the first `length` tokens of the cache. */
  async truncate(length: number): Promise<void> {
    if (length >= this.#tokens.length) return;
    if (length <= 0 || this.recurrent) {
      this.reset();
      return;
    }
    for (const [name, tensor] of this.#past) {
      const data = (tensor.location && tensor.location !== "cpu" && tensor.getData ? await tensor.getData() : tensor.data) as KvData;
      const [batch, heads, steps, dim] = tensor.dims as [number, number, number, number];
      const Ctor = data.constructor as new (n: number) => KvData;
      const out = new Ctor(batch * heads * length * dim);
      for (let b = 0; b < batch; b += 1) {
        for (let h = 0; h < heads; h += 1) {
          const from = ((b * heads + h) * steps) * dim;
          out.set(data.subarray(from, from + length * dim), ((b * heads + h) * length) * dim);
        }
      }
      tensor.dispose?.();
      this.#past.set(name, new this.Tensor(tensor.type, out, [batch, heads, length, dim]));
    }
    this.#tokens = this.#tokens.slice(0, length);
  }

  /** The cache on the host: per `past_key_values.*` input, its geometry, dtype and values. */
  async snapshot(): Promise<{ tokens: number[]; tensors: Map<string, { dims: number[]; type: string; data: KvData }> }> {
    const tensors = new Map<string, { dims: number[]; type: string; data: KvData }>();
    for (const [name, tensor] of this.#past) {
      const data = (tensor.location && tensor.location !== "cpu" && tensor.getData ? await tensor.getData() : tensor.data) as KvData;
      tensors.set(name, { dims: [...tensor.dims], type: tensor.type, data: data.slice() as KvData });
    }
    return { tokens: [...this.#tokens], tensors };
  }

  /** Replace the cache with a snapshot taken from the same model. */
  restore(tokens: readonly number[], tensors: ReadonlyMap<string, { dims: readonly number[]; type: string; data: KvData }>): void {
    const expected = this.session.inputs.filter((s) => s.name.startsWith("past_")).map((s) => s.name);
    const missing = expected.filter((name) => !tensors.has(name));
    if (missing.length > 0) throw new GenerationError(`the cache lacks ${missing.slice(0, 3).join(", ")}${missing.length > 3 ? "…" : ""}`);
    for (const [name, value] of tensors) {
      if (name.startsWith(PAST) && value.dims[2] !== tokens.length) throw new GenerationError(`${name} holds ${value.dims[2]} positions for ${tokens.length} tokens`);
    }
    this.reset();
    for (const name of expected) {
      const value = tensors.get(name)!;
      this.#past.set(name, new this.Tensor(value.type, value.data, [...value.dims]));
    }
    this.#tokens = [...tokens];
  }

  /** Forget everything. */
  reset(): void {
    for (const tensor of this.#past.values()) tensor.dispose?.();
    this.#past.clear();
    this.#tokens = [];
  }

  async close(): Promise<void> {
    this.reset();
    await this.bundle.close();
    await this.embed?.bundle.close();
  }
}

/** The logits of the last position, widened to float32. */
async function lastRow(tensor: TensorLike): Promise<Float32Array> {
  const data = tensor.location && tensor.location !== "cpu" && tensor.getData ? await tensor.getData() : tensor.data;
  const dims = tensor.dims;
  const vocab = dims[dims.length - 1]!;
  const total = (data as ArrayLike<number>).length;
  const from = total - vocab;
  if (tensor.type === "float16") {
    const Float16 = (globalThis as { Float16Array?: new (...args: never[]) => unknown }).Float16Array;
    if (Float16 && data instanceof Float16) return Float32Array.from((data as unknown as ArrayLike<number> & { subarray(a: number, b: number): ArrayLike<number> }).subarray(from, total));
    const half = data as Uint16Array;
    const out = new Float32Array(vocab);
    for (let i = 0; i < vocab; i += 1) out[i] = halfToFloat(half[from + i]!);
    return out;
  }
  return Float32Array.from((data as Float32Array).subarray(from, total));
}

function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}
