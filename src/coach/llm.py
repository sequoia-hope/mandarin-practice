"""LLM abstraction supporting Claude, Ollama, and Transformers backends."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""
    id: str
    name: str
    parameters: dict


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str  # Text response
    tool_calls: list[ToolCall]  # Tool calls to execute (if any)
    stop_reason: str  # "end_turn", "tool_use", etc.


class LLMBackend(ABC):
    """Abstract LLM backend interface."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> LLMResponse:
        """
        Send a chat request to the LLM.

        Args:
            messages: Conversation history
            system: System prompt
            tools: Tool definitions

        Returns:
            LLMResponse with content and/or tool calls
        """
        pass

    @abstractmethod
    def add_tool_result(
        self,
        messages: list[dict],
        tool_call_id: str,
        result: dict,
    ) -> list[dict]:
        """Add a tool result to the message history."""
        pass


class ClaudeBackend(LLMBackend):
    """Claude API backend via Anthropic SDK."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def chat(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> LLMResponse:
        # Convert tools to Claude format
        claude_tools = self._convert_tools(tools)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=claude_tools if claude_tools else None,
        )

        # Parse response
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    parameters=block.input,
                ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
        )

    def add_tool_result(
        self,
        messages: list[dict],
        tool_call_id: str,
        result: dict,
    ) -> list[dict]:
        """Add tool result in Claude format."""
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": json.dumps(result),
            }]
        })
        return messages

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert our tool format to Claude's format."""
        claude_tools = []
        for tool in tools:
            claude_tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": {
                    "type": "object",
                    "properties": tool.get("parameters", {}),
                    "required": tool.get("required", []),
                }
            })
        return claude_tools


class OllamaBackend(LLMBackend):
    """
    Local LLM via Ollama.

    Optimized for function calling with Qwen2.5, Llama 3.1, and other models.
    """

    # Models known to support function calling well
    FUNCTION_CALLING_MODELS = {
        "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5:72b",
        "llama3.1:8b", "llama3.1:70b",
        "mistral:7b", "mixtral:8x7b",
    }

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        try:
            import ollama
        except ImportError:
            raise ImportError("pip install ollama")

        self.client = ollama.Client(host=host, timeout=timeout)
        self.model = model

        # Check if model supports native function calling
        self._supports_tools = self._check_tool_support()

    def _check_tool_support(self) -> bool:
        """Check if model supports native tool calling."""
        # Qwen2.5 and Llama 3.1 support tools natively
        model_base = self.model.split(":")[0].lower()
        return any(
            known.startswith(model_base)
            for known in self.FUNCTION_CALLING_MODELS
        )

    def chat(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> LLMResponse:
        # Prepend system message
        full_messages = [{"role": "system", "content": system}] + messages

        if self._supports_tools and tools:
            # Use native tool calling
            return self._chat_with_tools(full_messages, tools)
        else:
            # Use prompt-based tool calling
            return self._chat_with_prompt_tools(full_messages, system, tools)

    def _chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> LLMResponse:
        """Chat using native Ollama tool calling."""
        ollama_tools = self._convert_tools(tools) if tools else None

        response = self.client.chat(
            model=self.model,
            messages=messages,
            tools=ollama_tools,
        )

        # Parse response
        message = response["message"]
        content = message.get("content", "")
        tool_calls = []

        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                # Handle different tool call formats
                if isinstance(tc, dict):
                    if "function" in tc:
                        # Standard format
                        tool_calls.append(ToolCall(
                            id=tc.get("id", f"call_{len(tool_calls)}"),
                            name=tc["function"]["name"],
                            parameters=tc["function"].get("arguments", {}),
                        ))
                    elif "name" in tc:
                        # Simplified format
                        tool_calls.append(ToolCall(
                            id=f"call_{len(tool_calls)}",
                            name=tc["name"],
                            parameters=tc.get("arguments", tc.get("parameters", {})),
                        ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )

    def _chat_with_prompt_tools(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> LLMResponse:
        """Chat using prompt-embedded tool instructions."""
        # Build enhanced system prompt with tool instructions
        enhanced_system = self._build_system_with_tools(system, tools)
        messages[0] = {"role": "system", "content": enhanced_system}

        response = self.client.chat(
            model=self.model,
            messages=messages,
        )

        content = response["message"].get("content", "")

        # Parse for tool calls in response
        tool_calls = self._parse_tool_calls(content)

        return LLMResponse(
            content=content if not tool_calls else "",
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )

    def _build_system_with_tools(self, system: str, tools: list[dict]) -> str:
        """Embed tool descriptions in system prompt."""
        if not tools:
            return system

        tool_desc = "\n\n## Available Tools\n\n"
        tool_desc += "You can use these tools by responding with a JSON object in this format:\n"
        tool_desc += '```json\n{"tool": "tool_name", "params": {...}}\n```\n\n'

        for tool in tools:
            tool_desc += f"### {tool['name']}\n"
            tool_desc += f"{tool['description']}\n"
            if tool.get("parameters"):
                tool_desc += "Parameters:\n"
                for param, info in tool["parameters"].items():
                    required = " (required)" if param in tool.get("required", []) else ""
                    tool_desc += f"  - {param}: {info.get('description', '')}{required}\n"
            tool_desc += "\n"

        return system + tool_desc

    def add_tool_result(
        self,
        messages: list[dict],
        tool_call_id: str,
        result: dict,
    ) -> list[dict]:
        """Add tool result in Ollama format."""
        messages.append({
            "role": "tool",
            "content": json.dumps(result),
        })
        return messages

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert to Ollama/OpenAI tool format."""
        ollama_tools = []
        for tool in tools:
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": tool.get("parameters", {}),
                        "required": tool.get("required", []),
                    }
                }
            })
        return ollama_tools

    def _parse_tool_calls(self, text: str) -> list[ToolCall]:
        """Parse tool calls from response text."""
        import re

        tool_calls = []

        # Look for JSON tool calls in various formats
        patterns = [
            r'\{[^{}]*"tool"[^{}]*\}',  # {"tool": "..."}
            r'```json\s*(\{[^`]*\})\s*```',  # ```json {...}```
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # Handle both raw match and group match
                    json_str = match if isinstance(match, str) else match[0]
                    parsed = json.loads(json_str)

                    if "tool" in parsed:
                        tool_calls.append(ToolCall(
                            id=f"call_{len(tool_calls)}",
                            name=parsed["tool"],
                            parameters=parsed.get("params", parsed.get("parameters", {})),
                        ))
                except json.JSONDecodeError:
                    continue

        return tool_calls


class TransformersBackend(LLMBackend):
    """Local LLM via HuggingFace Transformers (for models without Ollama)."""

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
    ):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("pip install transformers")

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map=device,
            torch_dtype="auto",
        )

    def chat(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> LLMResponse:
        # Build prompt with system and tools
        full_messages = [{"role": "system", "content": self._build_system_with_tools(system, tools)}]
        full_messages.extend(messages)

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
        )
        response_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # Parse for tool calls (simple JSON detection)
        tool_calls = self._parse_tool_calls(response_text)

        return LLMResponse(
            content=response_text if not tool_calls else "",
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )

    def add_tool_result(
        self,
        messages: list[dict],
        tool_call_id: str,
        result: dict,
    ) -> list[dict]:
        messages.append({
            "role": "user",
            "content": f"Tool result: {json.dumps(result)}",
        })
        return messages

    def _build_system_with_tools(self, system: str, tools: list[dict]) -> str:
        """Embed tool descriptions in system prompt for models without native tool support."""
        if not tools:
            return system

        tool_desc = "\n\nYou have access to these tools:\n"
        for tool in tools:
            tool_desc += f"\n- {tool['name']}: {tool['description']}\n"
            tool_desc += f"  Parameters: {json.dumps(tool.get('parameters', {}))}\n"

        tool_desc += "\nTo use a tool, respond with JSON: {\"tool\": \"name\", \"params\": {...}}"

        return system + tool_desc

    def _parse_tool_calls(self, text: str) -> list[ToolCall]:
        """Try to parse tool calls from response text."""
        import re

        tool_calls = []

        # Look for JSON tool calls
        json_pattern = r'\{[^{}]*"tool"[^{}]*\}'
        matches = re.findall(json_pattern, text)

        for i, match in enumerate(matches):
            try:
                parsed = json.loads(match)
                if "tool" in parsed:
                    tool_calls.append(ToolCall(
                        id=f"call_{i}",
                        name=parsed["tool"],
                        parameters=parsed.get("params", {}),
                    ))
            except json.JSONDecodeError:
                continue

        return tool_calls


def create_backend(
    backend_type: str = "ollama",
    **kwargs,
) -> LLMBackend:
    """
    Factory function to create LLM backend.

    Args:
        backend_type: "claude", "ollama", or "transformers"
        **kwargs: Backend-specific arguments

    Returns:
        LLMBackend instance
    """
    if backend_type == "claude":
        return ClaudeBackend(**kwargs)
    elif backend_type == "ollama":
        return OllamaBackend(**kwargs)
    elif backend_type == "transformers":
        return TransformersBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend_type}")


def check_ollama_available(model: str = "qwen2.5:7b") -> bool:
    """Check if Ollama is available and model is installed."""
    try:
        import ollama
        client = ollama.Client()
        models = client.list()
        model_names = [m.get("name", "") for m in models.get("models", [])]
        return any(model in name or name.startswith(model.split(":")[0]) for name in model_names)
    except Exception:
        return False
