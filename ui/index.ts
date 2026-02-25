import { LlamaIndexServer } from "@llamaindex/server";

new LlamaIndexServer({
  uiConfig: {
    llamaDeploy: { deployment: "chat", workflow: "workflow" },
  },
  llamaCloud: {
    outputDir: "output/llamacloud",
  },
}).start();
