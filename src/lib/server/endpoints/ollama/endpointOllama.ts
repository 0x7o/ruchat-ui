import {buildPrompt} from "$lib/buildPrompt";
import type {TextGenerationStreamOutput} from "@huggingface/inference";
import type {Endpoint} from "../endpoints";
import {z} from "zod";

export const endpointOllamaParametersSchema = z.object({
    weight: z.number().int().positive().default(1),
    model: z.any(),
    type: z.literal("ollama"),
    url: z.string().url().default("http://127.0.0.1:11434"),
    ollamaName: z.string().min(1).optional(),
});

export function endpointOllama(input: z.input<typeof endpointOllamaParametersSchema>): Endpoint {
    const {url, model, ollamaName} = endpointOllamaParametersSchema.parse(input);

    return async ({conversation}) => {
        const prompt = await buildPrompt({
            messages: conversation.messages,
            webSearch: conversation.messages[conversation.messages.length - 1].webSearch,
            preprompt: conversation.preprompt,
            model,
        });

        const r = await fetch(`http://127.0.0.1:8000/v1/completions`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                "model": model.id ?? model.name,
                "prompt": prompt,
                "max_tokens": 256,
                "temperature": 0.2,
                "repetition_penalty": 1.2,
                "stream": true,
                "n": 1
            }),
        });

        if (!r.ok) {
            throw new Error(`Failed to generate text: ${await r.text()}`);
        }

        const encoder = new TextDecoderStream();
        const reader = r.body?.pipeThrough(encoder).getReader();

        return (async function* () {
            let generatedText = "";
            let tokenId = 0;
            let stop = false;
            while (!stop) {
                // read the stream and log the outputs to console
                const out = (await reader?.read()) ?? {done: false, value: undefined};
                // we read, if it's done we cancel

                //{"c":"{\"id\": \"cmpl-ad30ea97180f429fa13a77396d57d1de\", \"created\": 4184, \"model\": \"0x7194633/fialka-7B-v3\", \"choices\": [{\"index\": 0, \"text\": \"\", \"logprobs\": null, \"finish_reason\": \"stop\"}]}\n\ndata: {\"id\": \"cmpl-ad30ea97180f429fa13a77396d57d1de\", \"created\": 4184, \"model\": \"0x7194633/fialka-7B-v3\", \"choices\": [{\"index\": 0, \"text\": \"\", \"logprobs\": null, \"finish_reason\": \"stop\"}], \"usage\": {\"prompt_tokens\": 81, \"total_tokens\": 113, \"completion_tokens\": 32}}\n\ndata: [DONE]\n\n"}
                // {"c":"{\"id\": \"cmpl-e72e76d281af46f796ec8e9f6aa14689\", \"created\": 4184, \"model\": \"0x7194633/fialka-7B-v3\", \"choices\": [{\"index\": 0, \"text\": \"\", \"logprobs\": null, \"finish_reason\": \"stop\"}]}\n\ndata: {\"id\": \"cmpl-e72e76d281af46f796ec8e9f6aa14689\", \"created\": 4184, \"model\": \"0x7194633/fialka-7B-v3\", \"choices\": [{\"index\": 0, \"text\": \"\", \"logprobs\": null, \"finish_reason\": \"stop\"}], \"usage\": {\"prompt_tokens\": 428, \"total_tokens\": 676, \"completion_tokens\": 248}}\n\ndata: [DONE]\n\n"}
                if (out.done) {
                    reader?.cancel();
                    return;
                }

                if (!out.value) {
                    return;
                }

                let data = null;
                let c = out.value.replace("data: ", "");
                try {
                    data = JSON.parse(c).choices[0];
                } catch (e) {
                    console.log(JSON.stringify({"c": c}));
                    return;
                }
                if (!data.finish_reason) {
                    generatedText += data.text;

                    yield {
                        token: {
                            id: tokenId++,
                            text: data.text ?? "",
                            logprob: 0,
                            special: false,
                        },
                        generated_text: null,
                        details: null,
                    } satisfies TextGenerationStreamOutput;
                } else {
                    stop = true;
                    yield {
                        token: {
                            id: tokenId++,
                            text: data.text ?? "",
                            logprob: 0,
                            special: true,
                        },
                        generated_text: generatedText,
                        details: null,
                    } satisfies TextGenerationStreamOutput;
                }
            }
        })();
    };
}

export default endpointOllama;
