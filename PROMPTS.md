# Prompt templates

These are the templates used to format the conversation history for different models used in HuggingChat. Set them in your `.env.local` [like so](https://github.com/huggingface/chat-ui#chatprompttemplate).

## Llama 2

```env
<s>[INST] <<SYS>>\n{{preprompt}}\n<</SYS>>\n\n{{#each messages}}{{#ifUser}}{{content}} [/INST] {{/ifUser}}{{#ifAssistant}}{{content}} </s><s>[INST] {{/ifAssistant}}{{/each}}
```

## CodeLlama

```env
<s>[INST] <<SYS>>\n{{preprompt}}\n<</SYS>>\n\n{{#each messages}}{{#ifUser}}{{content}} [/INST] {{/ifUser}}{{#ifAssistant}}{{content}} </s><s>[INST] {{/ifAssistant}}{{/each}}
```

## Falcon

```env
System: {{preprompt}}\nUser:{{#each messages}}{{#ifUser}}{{content}}\nFalcon:{{/ifUser}}{{#ifAssistant}}{{content}}\nUser:{{/ifAssistant}}{{/each}}
```

## Mistral

```env
<s>{{#each messages}}{{#ifUser}}[INST] {{#if @first}}{{#if @root.preprompt}}{{@root.preprompt}}\n{{/if}}{{/if}} {{content}} [/INST]{{/ifUser}}{{#ifAssistant}}{{content}}</s> {{/ifAssistant}}{{/each}}
```

## Zephyr

```env
<|system|>\n{{preprompt}}</s>\n{{#each messages}}{{#ifUser}}<|user|>\n{{content}}</s>\n<|assistant|>\n{{/ifUser}}{{#ifAssistant}}{{content}}</s>\n{{/ifAssistant}}{{/each}}
```

## IDEFICS

```env
{{#each messages}}{{#ifUser}}User: {{content}}{{/ifUser}}<end_of_utterance>\nAssistant: {{#ifAssistant}}{{content}}\n{{/ifAssistant}}{{/each}}
```

## OpenChat

```env
<s>{{#each messages}}{{#ifUser}}GPT4 User: {{#if @first}}{{#if @root.preprompt}}{{@root.preprompt}}\n{{/if}}{{/if}}{{content}}<|end_of_turn|>GPT4 Assistant: {{/ifUser}}{{#ifAssistant}}{{content}}<|end_of_turn|>{{/ifAssistant}}{{/each}}
```

## Mixtral

```env
<s> {{#each messages}}{{#ifUser}}[INST]{{#if @first}}{{#if @root.preprompt}}{{@root.preprompt}}\n{{/if}}{{/if}} {{content}} [/INST]{{/ifUser}}{{#ifAssistant}} {{content}}</s> {{/ifAssistant}}{{/each}}
```
