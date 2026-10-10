/** LanguageModel — a text LLM in the browser. */

import { Template } from "@huggingface/jinja";
import { detectCapabilities, huggingFaceResolver, loadModel, type LoadOptions, type ModelBundle, type ModelResolver, type ModelSession } from "@youni/onnxweb-core";
import { ConfigurationError, GenerationError, ModelNotFoundError, Tokenizer, defaultLogger, type Logger, type TensorFactory } from "@youni/onnxweb-core/core";

import {
  CONTEXT_FORMAT,
  CACHE_VERSION,
  CacheMismatchError,
  buildCacheExport,
  deserializeCache,
  identityMismatch,
  readCacheExport,
  serializeCache,
  type ContextExport,
  type ExportedCache,
  type KvQuantization,
  type ModelIdentity,
} from "./cache.js";
import { DEFAULT_LLM, VARIANT_COMPUTE_TYPE, entryFor, kvBytesPerToken, type LlmEntry, type LlmVariant } from "./catalogue.js";
import { Sampler, type SamplingSettings } from "./sampling.js";
import { DecoderSession } from "./session.js";

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool" | string;
  content: string;
  [key: string]: unknown;
}

export interface GenerateOptions extends SamplingSettings {
  /** Tokens written at most. */
  maxNewTokens?: number;
  /** Strings that end the generation (not included in the text). */
  stop?: readonly string[];
  signal?: AbortSignal;
  /** A thinking model (Qwen3, Qwen3.5, MiniCPM5): the reasoning block on (default) or off. */
  thinking?: boolean;
  /**
   * The reasoning as it is written, apart: the stream itself then carries the answer only. Without
   * it, nothing of the reasoning is streamed (it is in the result's `reasoning`).
   */
  onReasoning?: (delta: string) => void;
  /** Extra variables for the chat template (tools, dates…). */
  templateVariables?: Record<string, unknown>;
}

export interface GenerationStats {
  promptTokens: number;
  /** Prompt tokens served from the previous turn's cache. */
  cachedTokens: number;
  generatedTokens: number;
  /** Time to the first generated token. */
  ttftMs: number;
  totalMs: number;
  tokensPerSecond: number;
}

export interface GenerationResult {
  text: string;
  /** Qwen3's `<think>` block, apart from the answer. */
  reasoning: string | null;
  finishReason: "stop" | "length" | "cancelled";
  stats: GenerationStats;
}

export interface LanguageModelOptions extends Omit<LoadOptions, "model" | "computeType" | "resolver"> {
  Tensor: TensorFactory;
  /** A catalogue key, a repository, or `auto` (the default model). */
  model?: string;
  /** `auto`: q4f16 on WebGPU with shader-f16; int8 (when it fits) on WebAssembly; else q4. */
  variant?: LlmVariant | "auto";
  endpoint?: string;
  resolver?: ModelResolver;
  logger?: Logger;
  /** Tokens per prefill run. */
  chunkSize?: number;
  /**
   * The context, in tokens (prompt and answer together). `auto` (the default): the model's own
   * limit, as far as its KV cache fits beside its weights — on WebAssembly, the 32-bit memory (4
   * GB) less the weights and a margin; on WebGPU, `KV_BUDGET_GPU_MB`. A number is taken as asked
   * (and still capped at the model's limit).
   */
  contextTokens?: number | "auto";
}

/** The KV cache a GPU context is sized for, in MB (no API tells the free video memory). */
export const KV_BUDGET_GPU_MB = 2048;
/**
 * WebAssembly (32-bit): the memory in all, and what runtime and activations keep besides weights
 * and cache.
 */
const WASM_MEMORY_MB = 4096;
const WASM_MARGIN_MB = 768;

/**
 * The context `auto` gives: the model's limit, or what the KV cache budget holds — whichever is
 * smaller; never under 2048 tokens.
 */
export function autoContextTokens(entry: LlmEntry | null, variant: string, backend: string): number {
  const limit = entry?.contextTokens ?? 4096;
  if (!entry) return limit;
  const bytesPerValue = variant === "q4f16" || variant === "fp16" ? 2 : 4;
  const perTokenMb = kvBytesPerToken(entry, bytesPerValue) / 2 ** 20;
  const weightsMb = entry.sizesMb[variant as LlmVariant] ?? 0;
  const budgetMb = backend === "wasm" ? WASM_MEMORY_MB - weightsMb - WASM_MARGIN_MB : KV_BUDGET_GPU_MB;
  const fits = perTokenMb > 0 ? Math.floor(budgetMb / perTokenMb) : limit;
  return Math.max(2048, Math.min(limit, fits));
}

/** The reasoning block's tags, as text (when the tokenizer has no token for them). */
const THINK = /^\s*<think>([\s\S]*?)<\/think>\s*/;

export class LanguageModel {
  readonly #logger: Logger;
  #template: Template | null;
  /** The last conversation and the prompt it rendered to, for `exportContext()`. */
  #messages: ChatMessage[] = [];
  #text = "";

  private constructor(
    readonly name: string,
    readonly entry: LlmEntry | null,
    readonly variant: LlmVariant | string,
    readonly tokenizer: Tokenizer,
    readonly session: DecoderSession,
    readonly eosIds: ReadonlySet<number>,
    readonly contextTokens: number,
    template: string | null,
    readonly specialTokens: Record<string, string>,
    logger: Logger,
  ) {
    this.#logger = logger;
    this.#template = template ? new Template(template) : null;
  }

  /** Resolve, download and load a model. */
  static async load(options: LanguageModelOptions): Promise<LanguageModel> {
    const name = !options.model || options.model === "auto" ? DEFAULT_LLM : options.model;
    const found = entryFor(name);
    const [key, entry] = found ?? [name, null];
    const repo = entry?.repo ?? (name.includes("/") ? name : null);
    if (!repo && !options.resolver) throw new ModelNotFoundError(`'${name}' is neither in the catalogue nor a repository`);
    const logger = options.logger ?? defaultLogger;
    if (entry?.hybrid && options.backends !== "wasm") {
      logger.warn(`${key} runs on WebAssembly: onnxruntime-web's WebGPU backend computes its decoder wrongly`);
      options = { ...options, backends: "wasm" };
    }
    const variant = options.variant && options.variant !== "auto" ? options.variant : await LanguageModel.autoVariant(entry, options.backends);
    const { Tensor, endpoint, resolver, model: _m, variant: _v, chunkSize, contextTokens, ...rest } = options;
    const hub = resolver ?? huggingFaceResolver({ repo: repo!, subfolder: "onnx", manifest: false, ...(endpoint ? { endpoint } : {}) });
    const computeType = VARIANT_COMPUTE_TYPE[variant as LlmVariant] ?? variant;
    const extraFiles = ["tokenizer_config.json", "generation_config.json", "chat_template.jinja", "special_tokens_map.json"];
    let embed: { bundle: ModelBundle; session: ModelSession } | null = null;
    let bundle: ModelBundle;
    if (entry?.hybrid) {
      const suffix = VARIANT_SUFFIX[variant as LlmVariant] ?? "";
      const embedBundle = await loadModel({ ...rest, logger, resolver: graphAlias(hub, entry.hybrid.embed, suffix), model: { name: `${repo}#embed`, computeType } });
      embed = { bundle: embedBundle, session: embedBundle.only };
      try {
        bundle = await loadModel({ extraFiles, ...rest, logger, resolver: graphAlias(hub, entry.hybrid.decoder, suffix), model: { name: `${repo}#decoder`, computeType } });
      } catch (error) {
        await embedBundle.close();
        throw error;
      }
    } else {
      bundle = await loadModel({ extraFiles, ...rest, logger, resolver: hub, model: { name: repo ?? key, computeType } });
    }
    const tokenizerJson = bundle.assetText("tokenizer.json");
    if (!tokenizerJson) throw new GenerationError(`${key} came without a tokenizer.json`);
    const configText = bundle.assetText("tokenizer_config.json");
    const tokenizer = Tokenizer.fromJson(tokenizerJson, configText);
    const config = configText ? (JSON.parse(configText) as Record<string, unknown>) : {};
    const generation = JSON.parse(bundle.assetText("generation_config.json") ?? "{}") as { eos_token_id?: number | number[] };
    const eos = new Set<number>([generation.eos_token_id ?? []].flat());
    const special: Record<string, string> = {};
    for (const field of ["bos_token", "eos_token", "pad_token", "unk_token"]) {
      const value = config[field];
      const text = typeof value === "string" ? value : value && typeof value === "object" ? (value as { content?: string }).content : undefined;
      if (text) special[field] = text;
    }
    if (eos.size === 0 && special["eos_token"]) {
      const id = tokenizer.tokenToId(special["eos_token"]);
      if (id !== null) eos.add(id);
    }
    const template = bundle.assetText("chat_template.jinja") ?? templateOf(config["chat_template"]);
    const session = new DecoderSession(bundle, Tensor, chunkSize ?? 512, embed);
    const limit = entry?.contextTokens ?? 4096;
    const context = contextTokens === undefined || contextTokens === "auto" ? autoContextTokens(entry, variant, bundle.backend) : Math.min(limit, Math.max(16, Math.trunc(contextTokens)));
    const model = new LanguageModel(key, entry, variant, tokenizer, session, eos, context, template, special, logger);
    logger.info?.(`${key} (${variant}) loaded on ${bundle.backend}, context ${context} tokens (model limit ${limit})`);
    return model;
  }

  /**
   * q4f16 where float16 shaders exist. On WebAssembly, int8 when it is published and fits (≤ 2 GB):
   * onnxruntime-web has integer kernels for it, while its 4-bit MatMulNBits dequantizes the weights
   * at every token — Qwen3 0.6B, 128 tokens on 4 threads: int8 7.4 tokens/s, q4 2.4 (2026-10-10).
   * q4 otherwise; what the repository has after that.
   */
  static async autoVariant(entry: LlmEntry | null, backends?: unknown): Promise<LlmVariant> {
    const sizes = entry?.sizesMb ?? { q4f16: 1, q4: 1 };
    const caps = await detectCapabilities();
    const wantsWasm = backends === "wasm" || (Array.isArray(backends) && backends[0] === "wasm") || !caps.webgpu;
    const f16 = caps.webgpu && !wantsWasm && (caps as { shaderF16?: boolean }).shaderF16 !== false && !entry?.float16Overflows;
    const int8Fits = (sizes.int8 ?? Infinity) <= CPU_INT8_MAX_MB;
    const order: LlmVariant[] = f16
      ? ["q4f16", "q4", "int8", "fp16"]
      : wantsWasm && int8Fits
        ? ["int8", "q4", "q4f16", "fp16"]
        : ["q4", "int8", "q4f16", "fp16"];
    const variant = order.find((v) => sizes[v] !== undefined);
    if (!variant) throw new ConfigurationError(`no published variant of this model fits (${Object.keys(sizes).join(", ")})`);
    return variant;
  }

  get backend(): string {
    return this.session.session.backend;
  }

  /**
   * KV cache memory for `tokens` tokens, in MB (float16 values unless the variant computes in
   * float32).
   */
  cacheMb(tokens: number): number | null {
    if (!this.entry) return null;
    return (kvBytesPerToken(this.entry, this.variant === "q4f16" || this.variant === "fp16" ? 2 : 4) * tokens) / 2 ** 20;
  }

  countTokens(text: string): number {
    return this.tokenizer.encode(text, { addSpecialTokens: false }).length;
  }

  /** The chat rendered as the model was trained to read it. */
  applyChatTemplate(messages: readonly ChatMessage[], options: { addGenerationPrompt?: boolean; thinking?: boolean; variables?: Record<string, unknown> } = {}): string {
    if (!this.#template) throw new ConfigurationError(`${this.name} ships no chat template; use generate() with a prompt of your own`);
    return this.#template.render({
      messages,
      add_generation_prompt: options.addGenerationPrompt ?? true,
      enable_thinking: options.thinking ?? true,
      ...this.specialTokens,
      ...(options.variables ?? {}),
    });
  }

  /** Continue a raw prompt. */
  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerationResult> {
    let result: GenerationResult | null = null;
    for await (const event of this.#run(this.tokenizer.encode(prompt, { addSpecialTokens: true }), options)) if (event.done) result = event.result;
    return result!;
  }

  /** Answer a conversation. */
  async chat(messages: readonly ChatMessage[], options: GenerateOptions = {}): Promise<GenerationResult> {
    let result: GenerationResult | null = null;
    for await (const event of this.#run(this.#chatIds(messages, options), options)) if (event.done) result = event.result;
    this.#messages = [...messages, { role: "assistant", content: result!.text }];
    return result!;
  }

  /** The text as it is written: a prompt (string) or a conversation. */
  async *stream(input: string | readonly ChatMessage[], options: GenerateOptions = {}): AsyncGenerator<string, GenerationResult> {
    const ids = typeof input === "string" ? this.tokenizer.encode(input, { addSpecialTokens: true }) : this.#chatIds(input, options);
    for await (const event of this.#run(ids, options)) {
      if (event.done) return event.result;
      yield event.delta;
    }
    throw new GenerationError("the generation ended without a result");
  }

  #chatIds(messages: readonly ChatMessage[], options: GenerateOptions): number[] {
    const text = this.applyChatTemplate(messages, { thinking: options.thinking ?? true, ...(options.templateVariables ? { variables: options.templateVariables } : {}) });
    this.#text = text;

    return this.tokenizer.encode(text, { addSpecialTokens: false });
  }

  async *#run(prompt: number[], options: GenerateOptions): AsyncGenerator<{ done: false; delta: string } | { done: true; result: GenerationResult }> {
    const started = performance.now();
    const asked = options.maxNewTokens ?? 512;
    if (prompt.length >= this.contextTokens) {
      throw new ConfigurationError(`${prompt.length} prompt tokens fill the ${this.contextTokens}-token context of ${this.name}`);
    }

    const maxNew = Math.min(asked, this.contextTokens - prompt.length);
    if (maxNew < asked) this.#logger.info?.(`${this.name}: at most ${maxNew} new tokens (${prompt.length} in the prompt, ${this.contextTokens} in the context)`);
    const sampler = new Sampler(options);
    const { logits: first, reused } = await this.session.prime(prompt, options.signal);
    let logits = first;
    const history = [...prompt];
    const generated: number[] = [];

    const open = this.tokenizer.tokenToId("<think>");
    const close = this.tokenizer.tokenToId("</think>");
    const tagged = typeof open === "number" && typeof close === "number";
    let thinking = tagged && prompt.lastIndexOf(open) > prompt.lastIndexOf(close);
    const answer: number[] = [];
    const thought: number[] = [];
    let thoughtShown = "";
    let emitted = "";
    let ttft = 0;
    let finish: GenerationResult["finishReason"] = "length";
    const stops = options.stop ?? [];
    try {
      while (generated.length < maxNew) {
        if (options.signal?.aborted) {
          finish = "cancelled";
          break;
        }
        const token = sampler.next(logits, history);
        if (generated.length === 0) ttft = performance.now() - started;
        if (this.eosIds.has(token)) {
          finish = "stop";
          break;
        }
        generated.push(token);
        history.push(token);
        if (tagged && (token === open || token === close)) {
          thinking = token === open;
          if (generated.length >= maxNew) break;
          logits = await this.session.feed([token], options.signal);
          continue;
        }
        if (thinking) {
          thought.push(token);
          if (options.onReasoning) {
            const shown = this.tokenizer.decode(thought, { skipSpecialTokens: true });
            const stable = shown.endsWith("\uFFFD") ? shown.slice(0, -1) : shown;
            if (stable.length > thoughtShown.length) {
              options.onReasoning(stable.slice(thoughtShown.length));
              thoughtShown = stable;
            }
          }
          if (generated.length >= maxNew) break;
          logits = await this.session.feed([token], options.signal);
          continue;
        }
        answer.push(token);
        let text = this.tokenizer.decode(tagged ? answer : generated, { skipSpecialTokens: true });
        const hit = stops.map((s) => text.indexOf(s, Math.max(0, emitted.length - s.length))).filter((i) => i >= 0);
        if (hit.length > 0) {
          text = text.slice(0, Math.min(...hit));
          if (text.length > emitted.length) yield { done: false, delta: text.slice(emitted.length) };
          emitted = text;
          finish = "stop";
          break;
        }

        const stable = text.endsWith("�") ? text.slice(0, -1) : text;
        const holdBack = Math.max(0, ...stops.map((s) => s.length - 1));
        const safe = stable.slice(0, Math.max(emitted.length, stable.length - holdBack));
        if (safe.length > emitted.length) {
          yield { done: false, delta: safe.slice(emitted.length) };
          emitted = safe;
        }
        if (generated.length >= maxNew) break;
        logits = await this.session.feed([token], options.signal);
      }
    } catch (error) {
      if ((error as { name?: string }).name === "CancelledError") finish = "cancelled";
      else throw error;
    }
    if (finish !== "stop" || stops.length === 0) {
      const final = this.tokenizer.decode(tagged ? answer : generated, { skipSpecialTokens: true });
      if (final.length > emitted.length && final.startsWith(emitted)) yield { done: false, delta: final.slice(emitted.length) };
      if (final.startsWith(emitted)) emitted = final;
    }
    const total = performance.now() - started;
    const decodeMs = Math.max(total - ttft, 1);

    const textual = tagged ? null : THINK.exec(emitted);
    const reasoning = tagged ? (thought.length > 0 ? this.tokenizer.decode(thought, { skipSpecialTokens: true }).trim() : null) : textual ? textual[1]!.trim() : null;
    yield {
      done: true,
      result: {
        text: (textual ? emitted.slice(textual[0].length) : emitted).trim(),
        reasoning: reasoning || null,
        finishReason: finish,
        stats: {
          promptTokens: prompt.length,
          cachedTokens: reused,
          generatedTokens: generated.length,
          ttftMs: Math.round(ttft),
          totalMs: Math.round(total),
          tokensPerSecond: generated.length > 1 ? Math.round(((generated.length - 1) / decodeMs) * 1000 * 10) / 10 : 0,
        },
      },
    };
  }

  /** Forget the cached conversation. */
  reset(): void {
    this.session.reset();
    this.#messages = [];
    this.#text = "";
  }

  /** What a cache from this model must match: catalogue key, variant, geometry, graph files. */
  get identity(): ModelIdentity {
    const keys = this.session.session.inputs.filter((s) => /^past_key_values\.\d+\.key$/.test(s.name));
    const dims = keys[0]?.dims ?? [];
    const numeric = (value: unknown, fallback: number) => (typeof value === "number" && value > 0 ? value : fallback);
    const graphs = this.session.bundle.manifest.graphs.map((g) => `${g.filename}:${g.sizeMb}`).join(",");
    return {
      model: this.name,
      repo: this.entry?.repo ?? (this.name.includes("/") ? this.name : null),
      variant: String(this.variant),
      layers: keys.length,
      kvHeads: numeric(dims[1], this.entry?.kvHeads ?? 0),
      headDim: numeric(dims[3], this.entry?.headDim ?? 0),
      fingerprint: graphs,
    };
  }

  /**
   * The KV cache as one buffer: a header naming this model, then keys and values layer by layer,
   * each optionally int8 at rest (keys per channel, values per token).
   */
  async exportCache(options: { key?: KvQuantization; value?: KvQuantization } = {}): Promise<Uint8Array> {
    return serializeCache(await this.exportCacheParts(options));
  }

  /** The export as header and body, for a caller that stores them apart. */
  async exportCacheParts(options: { key?: KvQuantization; value?: KvQuantization } = {}): Promise<ExportedCache> {
    if (this.session.length === 0) throw new ConfigurationError("the cache is empty: nothing has been generated yet");
    return buildCacheExport(this.identity, await this.session.snapshot(), options);
  }

  /**
   * Load a cache exported from THIS model; refused for any other, naming the one it needs (see
   * `loadForCache`). The next `chat()` or `generate()` reuses every token it holds.
   */
  importCache(cache: Uint8Array | ArrayBuffer | ExportedCache): void {
    const exported = cache instanceof Uint8Array || cache instanceof ArrayBuffer ? deserializeCache(cache) : cache;
    const differs = identityMismatch(exported.header.identity, this.identity);
    if (differs.length > 0) {
      const wanted = exported.header.identity;
      throw new CacheMismatchError(
        `this cache was computed by ${wanted.model} (${wanted.variant}), not ${this.name} (${this.variant}): ${differs.join(", ")} differ. Load the model it names (LanguageModel.loadForCache) to reuse it.`,
      );
    }
    this.session.restore(exported.header.tokens, readCacheExport(exported));
  }

  /**
   * The conversation and its tokens, as JSON: rebuilt by `importContext()` at the cost of a
   * prefill.
   */
  exportContext(): ContextExport {
    return { format: CONTEXT_FORMAT, version: CACHE_VERSION, identity: this.identity, messages: this.#messages.map((m) => ({ role: m.role, content: m.content })), tokens: [...this.session.tokens], text: this.#text };
  }

  /**
   * Restore a conversation: its messages, and its tokens recomputed into the cache (any model reads
   * a context; only the same one reuses a KV cache).
   */
  async importContext(context: ContextExport, options: { signal?: AbortSignal } = {}): Promise<void> {
    if (context.format !== CONTEXT_FORMAT) throw new ConfigurationError(`not a context export (format '${String(context.format)}')`);
    this.#messages = context.messages.map((m) => ({ ...m }));
    this.#text = context.text;
    const tokens = identityMismatch(context.identity, this.identity).length === 0 ? context.tokens : this.tokenizer.encode(context.text, { addSpecialTokens: false });
    if (tokens.length > 0) await this.session.prime(tokens, options.signal);
  }

  /** The conversation `chat()` last answered, its answer included. */
  get messages(): readonly ChatMessage[] {
    return this.#messages;
  }

  /** Load the very model a cache was computed by, then the cache into it. */
  static async loadForCache(cache: Uint8Array | ArrayBuffer | ExportedCache, options: Omit<LanguageModelOptions, "model" | "variant">): Promise<LanguageModel> {
    const exported = cache instanceof Uint8Array || cache instanceof ArrayBuffer ? deserializeCache(cache) : cache;
    const { identity } = exported.header;
    const model = await LanguageModel.load({ ...options, model: identity.repo ?? identity.model, variant: identity.variant as LlmVariant });
    try {
      model.importCache(exported);
    } catch (error) {
      await model.close();
      throw error;
    }
    return model;
  }

  async close(): Promise<void> {
    await this.session.close();
    this.#template = null;
    this.#logger.debug?.(`${this.name} closed`);
  }
}

/** `chat_template` as tokenizer_config.json gives it: a string, or named templates. */
function templateOf(value: unknown): string | null {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    const named = value as { name?: string; template?: string }[];
    return (named.find((t) => t.name === "default") ?? named[0])?.template ?? null;
  }
  return null;
}

/** File suffix of each variant in the published naming convention. */
/** The largest int8 export `autoVariant` takes on WebAssembly (32-bit memory: 4 GB in all). */
const CPU_INT8_MAX_MB = 2000;

const VARIANT_SUFFIX: Readonly<Record<LlmVariant, string>> = { q4f16: "_q4f16", q4: "_q4", int8: "_quantized", fp16: "_fp16" };

/**
 * A source that shows one graph of a repository as its only model (`model{suffix}.onnx`), its
 * external data under their real names (the graph records them). A hybrid model's embedding and
 * decoder graphs are not an encoder-decoder pair, and the published conventions have no name for
 * them.
 */
function graphAlias(base: ModelResolver, stem: string, suffix: string): ModelResolver {
  const alias = `model${suffix}.onnx`;
  const real = `${stem}${suffix}.onnx`;
  return {
    id: `${base.id}#${stem}`,
    manifestUrls: () => [],
    assetUrl: (key, filename, place) => {
      const asset = base.assetUrl(key, filename === alias ? real : filename, place);
      return { ...asset, filename };
    },
    listFiles: async (key, signal) => {
      const listing = await base.listFiles?.(key, signal);
      if (!listing) return null;
      const graphs = listing.graphs
        .filter((f) => f.name === real || f.name.startsWith(`${real}_data`) || f.name.startsWith(`${real}.data`))
        .map((f) => (f.name === real ? { ...f, name: alias } : f));
      return { graphs, root: listing.root };
    },
  };
}
