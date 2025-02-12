import requests
from langchain_core.language_models.llms import LLM


class MyHostedLLM(LLM):
    url: str

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        messages = self.format_prompt(prompt)
        payload = {"model": "deepseek-r1:32b", "messages": messages, "stream": False}
        response = requests.post(self.url, json=payload, timeout=90)
        response.raise_for_status()
        json_text = response.json()
        return json_text["message"]["content"].split("</think>\n\n")[1]

    def format_prompt(self, prompt_str):
        messages = []
        prompt = {}
        i = prompt_str.find("System")
        prompt_str = prompt_str[i + 8 :]

        i = prompt_str.find("Human:")
        prompt_system, prompt_user = prompt_str[:i], prompt_str[i:]

        prompt["role"] = "System"
        prompt["content"] = prompt_system
        messages.append(prompt)

        prompt = {}
        prompt["role"] = "User"
        prompt["content"] = prompt_user
        messages.append(prompt)
        return messages

    @property
    def _llm_type(self) -> str:
        return "my_hosted_llm"
