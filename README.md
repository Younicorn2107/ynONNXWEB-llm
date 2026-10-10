# onnxweb-llm

Language models in the browser: chat and streaming, the KV cache kept on the GPU.

Decoder-only ONNX exports (Qwen3, Qwen3.5, Llama 3.2, Gemma 3, SmolLM2, Qwen2.5): q4f16 on WebGPU, int8 or q4 on WebAssembly, chosen automatically; the context length sized to the device's memory; the cache reused from one turn to the next; the model's own chat template.

```ts
import * as ort from "onnxruntime-web";
import { LanguageModel } from "@youni/onnxweb-llm";

const llm = await LanguageModel.load({ ort, Tensor: ort.Tensor, model: "qwen3-0.6b" });
const reply = await llm.chat([{ role: "user", content: "Bonjour !" }], { thinking: false });
for await (const piece of llm.stream("Once upon a time", { maxNewTokens: 64 })) console.log(piece);
```

Weights: Qwen and SmolLM2 Apache-2.0; Llama 3.2 Community License ("Built with Llama"); Gemma Terms of Use.

## Install

Builds against `@youni/onnxweb-core`:

```sh
npm install
npm run build   # dist/
npm test        # unit tests (Node, onnxruntime-web on WebAssembly)
```

## Licence

Apache-2.0 for the code. Model weights keep their own licence (listed above where it matters).
