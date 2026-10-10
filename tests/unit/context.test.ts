import { describe, expect, it } from "vitest";

import { LLM_CATALOGUE } from "../../src/catalogue.js";
import { autoContextTokens } from "../../src/model.js";

describe("the context auto gives", () => {
  const qwen = LLM_CATALOGUE["qwen3-0.6b"]!;

  it("on WebAssembly, what the 32-bit memory holds beside the weights", () => {
    expect(autoContextTokens(qwen, "q4", "wasm")).toBe(Math.floor((4096 - 919 - 768) / (229376 / 2 ** 20)));
  });

  it("on WebGPU, what the GPU budget holds, float16 values", () => {
    expect(autoContextTokens(qwen, "q4f16", "webgpu")).toBe(Math.floor(2048 / (114688 / 2 ** 20)));
  });

  it("never more than the model's limit, never under 2048", () => {
    const small = LLM_CATALOGUE["smollm2-360m"]!;
    expect(autoContextTokens(small, "q4f16", "webgpu")).toBe(small.contextTokens);
    expect(autoContextTokens(LLM_CATALOGUE["llama-3.2-3b"]!, "q4", "wasm")).toBe(2048);
  });
});
