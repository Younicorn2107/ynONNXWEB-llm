/**
 * The LLM catalogue: decoder-only ONNX exports (GroupQueryAttention, MatMulNBits) a browser can
 * fetch. Every row was read from the Hub on 2026-10-05 — the `config.json` for the geometry, the
 * `onnx/` tree for the sizes (graph plus every external-data shard, in MB).
 */

export type LlmVariant = "q4f16" | "q4" | "int8" | "fp16";

export interface LlmEntry {
  repo: string;
  /** The model the export was made from. */
  baseModel: string;
  family: string;
  layers: number;
  kvHeads: number;
  heads: number;
  headDim: number;
  /** Positions the model was trained for. */
  contextTokens: number;
  /** Download size per variant, in MB (graph + external data). */
  sizesMb: Partial<Record<LlmVariant, number>>;
  /** Parameters, in millions. */
  paramsM: number;
  licence: string;
  /** Qwen3's `enable_thinking`: the template can switch the reasoning block on and off. */
  thinking: boolean;
  notes: string | null;
  /**
   * A hybrid model (Qwen3.5): token embeddings in a graph of their own, and a decoder that also
   * carries linear-attention states. The file stems of both graphs, and how many layers keep a
   * growing KV cache.
   */
  hybrid?: { embed: string; decoder: string; kvLayers: number };
  /**
   * Its float16 graphs overflow on WebGPU (garbled text): `auto` takes the float32-activation q4
   * instead. Qwen2.5 0.5B, onnxruntime-web 1.30.
   */
  float16Overflows?: boolean;
}

export const LLM_CATALOGUE: Readonly<Record<string, LlmEntry>> = {
  "qwen3-0.6b": {
    repo: "onnx-community/Qwen3-0.6B-ONNX",
    baseModel: "Qwen/Qwen3-0.6B",
    family: "qwen3",
    layers: 28, kvHeads: 8, heads: 16, headDim: 128, contextTokens: 40960,
    sizesMb: { q4f16: 570, q4: 919, int8: 618, fp16: 1203 },
    paramsM: 600, licence: "apache-2.0", thinking: true,
    notes: "The default: small enough for WASM in q4, and a thinking mode.",
  },
  "qwen3-1.7b": {
    repo: "onnx-community/Qwen3-1.7B-ONNX",
    baseModel: "Qwen/Qwen3-1.7B",
    family: "qwen3",
    layers: 28, kvHeads: 8, heads: 16, headDim: 128, contextTokens: 40960,
    sizesMb: { q4f16: 1426, q4: 2147, int8: 1742, fp16: 3452 },
    paramsM: 1700, licence: "apache-2.0", thinking: true,
    notes: null,
  },
  "qwen3-4b": {
    repo: "onnx-community/Qwen3-4B-ONNX",
    baseModel: "Qwen/Qwen3-4B",
    family: "qwen3",
    layers: 36, kvHeads: 8, heads: 32, headDim: 128, contextTokens: 40960,
    sizesMb: { q4f16: 2833, fp16: 8055 },
    paramsM: 4000, licence: "apache-2.0", thinking: true,
    notes: "WebGPU only: no q4 graph for WASM, and q4f16 alone is 2.8 GB.",
  },
  "qwen3.5-0.8b": {
    repo: "onnx-community/Qwen3.5-0.8B-ONNX",
    baseModel: "Qwen/Qwen3.5-0.8B",
    family: "qwen3.5",
    layers: 24, kvHeads: 2, heads: 8, headDim: 256, contextTokens: 262144,
    sizesMb: { q4f16: 584, q4: 648, int8: 1167, fp16: 2015 },
    paramsM: 800, licence: "apache-2.0", thinking: true,
    notes: "Hybrid: three layers in four use linear attention (fixed-size state), so the KV cache grows on 6 layers only. Text only here (the vision encoder is not loaded).",
    hybrid: { embed: "embed_tokens", decoder: "decoder_model_merged", kvLayers: 6 },
  },
  "qwen3.5-2b": {
    repo: "onnx-community/Qwen3.5-2B-ONNX",
    baseModel: "Qwen/Qwen3.5-2B",
    family: "qwen3.5",
    layers: 24, kvHeads: 2, heads: 8, headDim: 256, contextTokens: 262144,
    sizesMb: { q4f16: 1384, q4: 1535, int8: 2767, fp16: 4781 },
    paramsM: 2000, licence: "apache-2.0", thinking: true,
    notes: "Hybrid (linear attention on 18 of 24 layers). Text only here.",
    hybrid: { embed: "embed_tokens", decoder: "decoder_model_merged", kvLayers: 6 },
  },
  "qwen3.5-4b": {
    repo: "onnx-community/Qwen3.5-4B-ONNX",
    baseModel: "Qwen/Qwen3.5-4B",
    family: "qwen3.5",
    layers: 32, kvHeads: 4, heads: 16, headDim: 256, contextTokens: 262144,
    sizesMb: { q4f16: 2803, q4: 3107, int8: 5604, fp16: 9685 },
    paramsM: 4000, licence: "apache-2.0", thinking: true,
    notes: "Hybrid (linear attention on 24 of 32 layers). WebGPU in practice: 2.8 GB at its lightest. Text only here.",
    hybrid: { embed: "embed_tokens", decoder: "decoder_model_merged", kvLayers: 8 },
  },
  "minicpm5-2b": {
    repo: "Mike0021/MiniCPM5-2B-ONNX",
    baseModel: "openbmb/MiniCPM5-2B",
    family: "llama",
    layers: 42, kvHeads: 2, heads: 16, headDim: 128, contextTokens: 131072,
    sizesMb: { q4f16: 1832 },
    paramsM: 2000, licence: "apache-2.0", thinking: true,
    notes: "A community export (q4f16 only: WebGPU with shader-f16). Two KV heads: a small cache per token.",
  },
  "llama-3.2-1b": {
    repo: "onnx-community/Llama-3.2-1B-Instruct-ONNX",
    baseModel: "meta-llama/Llama-3.2-1B-Instruct",
    family: "llama",
    layers: 16, kvHeads: 8, heads: 32, headDim: 64, contextTokens: 131072,
    sizesMb: { q4f16: 1090, q4: 1693, int8: 1237, fp16: 2488 },
    paramsM: 1240, licence: "llama3.2", thinking: false,
    notes: "Llama 3.2 Community License: acceptable-use policy, attribution.",
  },
  "llama-3.2-3b": {
    repo: "onnx-community/Llama-3.2-3B-Instruct-ONNX",
    baseModel: "meta-llama/Llama-3.2-3B-Instruct",
    family: "llama",
    layers: 28, kvHeads: 8, heads: 24, headDim: 128, contextTokens: 131072,
    sizesMb: { q4f16: 2407, q4: 3405, int8: 3214, fp16: 6460 },
    paramsM: 3210, licence: "llama3.2", thinking: false,
    notes: null,
  },
  "gemma-3-1b": {
    repo: "onnx-community/gemma-3-1b-it-ONNX",
    baseModel: "google/gemma-3-1b-it",
    family: "gemma3",
    layers: 26, kvHeads: 1, heads: 4, headDim: 256, contextTokens: 32768,
    sizesMb: { q4f16: 763, q4: 859, int8: 1526, fp16: 2034 },
    paramsM: 1000, licence: "gemma", thinking: false,
    notes: "Gemma Terms of Use. One KV head: the smallest cache per token here.",
  },
  "smollm2-360m": {
    repo: "HuggingFaceTB/SmolLM2-360M-Instruct",
    baseModel: "HuggingFaceTB/SmolLM2-360M-Instruct",
    family: "llama",
    layers: 32, kvHeads: 5, heads: 15, headDim: 64, contextTokens: 8192,
    sizesMb: { q4f16: 273, q4: 388, int8: 365, fp16: 725 },
    paramsM: 360, licence: "apache-2.0", thinking: false,
    notes: "The smallest: for a quick test, or a phone.",
  },
  "qwen2.5-0.5b": {
    repo: "onnx-community/Qwen2.5-0.5B-Instruct",
    baseModel: "Qwen/Qwen2.5-0.5B-Instruct",
    family: "qwen2",
    layers: 24, kvHeads: 2, heads: 14, headDim: 64, contextTokens: 32768,
    sizesMb: { q4f16: 483, q4: 786, int8: 512, fp16: 997 },
    paramsM: 494, licence: "apache-2.0", thinking: false,
    float16Overflows: true,
    notes: null,
  },
};

export const DEFAULT_LLM = "qwen3-0.6b";

/**
 * The compute type each variant loads as, in the core's naming (`model_q4f16.onnx` is
 * `int4_float16`).
 */
export const VARIANT_COMPUTE_TYPE: Readonly<Record<LlmVariant, string>> = {
  q4f16: "int4_float16",
  q4: "int4",
  int8: "int8",
  fp16: "float16",
};

/** KV cache bytes per token: layers × 2 (key, value) × kvHeads × headDim × bytes per value. */
export function kvBytesPerToken(entry: LlmEntry, bytesPerValue = 2): number {
  if (entry.hybrid) return 2 * entry.hybrid.kvLayers * entry.kvHeads * entry.headDim * bytesPerValue;
  return entry.layers * 2 * entry.kvHeads * entry.headDim * bytesPerValue;
}

export function entryFor(name: string): [string, LlmEntry] | null {
  const key = name.trim().toLowerCase();
  if (LLM_CATALOGUE[key]) return [key, LLM_CATALOGUE[key]!];
  const byRepo = Object.entries(LLM_CATALOGUE).find(([, e]) => e.repo.toLowerCase() === key || e.baseModel.toLowerCase() === key);
  return byRepo ?? null;
}
