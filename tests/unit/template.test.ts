import { describe, expect, it } from "vitest";

import { ConfigurationError } from "@youni/onnxweb-core/core";

import { ChatTemplate, MAX_TEMPLATE_CHARS } from "../../src/template.js";

describe("a chat template from a repository", () => {
  it("renders a real one as before, range() included", () => {
    const template = new ChatTemplate("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}{% for i in range(3) %}{{ i }}{% endfor %}{% for i in range(1, 7, 2) %}{{ i }}{% endfor %}");
    expect(template.render({ messages: [{ role: "user", content: "hi" }] })).toBe("<|user|>hi012135");
  });

  it("refuses a loop that would freeze the page, quickly", () => {
    const started = Date.now();
    for (const loop of ["{% for i in range(10**9) %}x{% endfor %}", "{% for i in range(1000) %}{% for j in range(1000) %}x{% endfor %}{% endfor %}"]) {
      expect(() => new ChatTemplate(loop).render({})).toThrow(ConfigurationError);
    }
    expect(Date.now() - started).toBeLessThan(2000);
  });

  it("refuses a template longer than any published one", () => {
    expect(() => new ChatTemplate("x".repeat(MAX_TEMPLATE_CHARS + 1))).toThrow(ConfigurationError);
  });
});
