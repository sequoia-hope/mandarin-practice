"""Main conversational tone coach."""

import json
from typing import Optional

from .llm import LLMBackend, LLMResponse, create_backend
from .tools import COACH_TOOLS, ToolExecutor


# System prompt optimized for both Claude and local LLMs
SYSTEM_PROMPT = """You are a friendly Mandarin Chinese tone coach helping users learn correct tone pronunciation.

## Your Role
- Help users learn to pronounce Chinese tones correctly
- Provide translations when users ask "How do I say X?"
- Guide practice with audio and evaluation
- Give encouraging, specific feedback

## Available Tools

You have these tools:
1. **generate_audio** - Create native speaker audio for Chinese text
2. **record_and_evaluate** - Record user's speech and check their tones
3. **play_audio** - Replay audio files
4. **show_pitch_comparison** - Show visual pitch comparison

## How to Help

### When user asks "How do I say X?"
1. Provide the translation with:
   - Chinese characters
   - Pinyin with tone marks (e.g., nǐ hǎo)
   - Tone numbers (e.g., T3 T3)
   - English meaning

2. Use generate_audio to create reference audio

3. Use record_and_evaluate to check their pronunciation

### When giving feedback:
- Celebrate correct tones!
- For mistakes: "You said T2 (rising) but it should be T3 (dipping)"
- Give tips: "Try starting lower and going down before up"
- Offer to try again

## Tone Quick Reference
- T1 (ˉ): High level - like singing a high note "laaaa"
- T2 (ˊ): Rising - like asking "what?"
- T3 (ˇ): Dipping - go down then up, like "well..."
- T4 (ˋ): Falling - sharp drop, like "stop!"
- T5: Neutral - light and quick

## Guidelines
- Keep responses concise
- Always use tools for audio/evaluation - don't pretend to hear
- Break long phrases into 2-4 syllable chunks
- Be encouraging!

Now help the user learn Mandarin tones!"""


# Simplified prompt for local LLMs (more structured, less verbose)
SYSTEM_PROMPT_LOCAL = """You are a Mandarin Chinese tone coach.

## Your Job
Help users learn Chinese tones through practice.

## Tools Available
- generate_audio: Create Chinese audio for text
- record_and_evaluate: Record and check user's pronunciation
- play_audio: Replay audio
- show_pitch_comparison: Show pitch visualization

## When User Asks "How do I say X?"
1. Give translation: characters, pinyin (with marks), tones
2. Call generate_audio for the Chinese text
3. Call record_and_evaluate to check their attempt
4. Give feedback on results

## Tone Reference
- T1: High level (sing "laaa")
- T2: Rising (ask "what?")
- T3: Dipping (say "well...")
- T4: Falling (command "stop!")
- T5: Neutral (light, quick)

## Feedback Tips
- Celebrate correct tones
- For errors: "You said T2 but should be T3"
- Offer retry

Keep responses short. Use tools for audio/evaluation."""


class ToneCoach:
    """Conversational tone coach with tool calling."""

    def __init__(
        self,
        llm_backend: LLMBackend,
        tool_executor: ToolExecutor,
        use_local_prompt: bool = False,
    ):
        """
        Initialize tone coach.

        Args:
            llm_backend: LLM backend for conversation
            tool_executor: Tool executor for audio/evaluation
            use_local_prompt: Use simplified prompt for local LLMs
        """
        self.llm = llm_backend
        self.tools = tool_executor
        self.conversation_history: list[dict] = []

        # Select appropriate system prompt
        self.system_prompt = SYSTEM_PROMPT_LOCAL if use_local_prompt else SYSTEM_PROMPT

    def chat(self, user_message: str) -> str:
        """
        Process user message and return response.

        Handles tool calling loop internally.
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        max_tool_rounds = 10  # Prevent infinite loops

        for _ in range(max_tool_rounds):
            try:
                # Get LLM response
                response = self.llm.chat(
                    messages=self.conversation_history,
                    system=self.system_prompt,
                    tools=COACH_TOOLS,
                )
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {e}"
                self.conversation_history.append({
                    "role": "assistant",
                    "content": error_msg
                })
                return error_msg

            # If there are tool calls, execute them
            if response.tool_calls:
                # Add assistant message with tool calls to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "parameters": tc.parameters}
                        for tc in response.tool_calls
                    ]
                })

                # Execute each tool and add results
                for tool_call in response.tool_calls:
                    try:
                        result = self.tools.execute(tool_call.name, tool_call.parameters)
                    except Exception as e:
                        result = {"error": str(e)}

                    self.llm.add_tool_result(
                        self.conversation_history,
                        tool_call.id,
                        result
                    )

                # Continue loop to process tool results
                continue
            else:
                # No tool calls - final response
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content
                })
                return response.content

        return "I got a bit confused. Let's start fresh - what would you like to learn?"

    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []


def create_coach(
    backend: str = "ollama",
    model: Optional[str] = None,
    tts=None,
    evaluator=None,
    **backend_kwargs,
) -> ToneCoach:
    """
    Create a ToneCoach with specified backend.

    Args:
        backend: "claude", "ollama", or "transformers"
        model: Model name (backend-specific)
        tts: TTS instance for audio generation
        evaluator: Evaluator instance for tone checking
        **backend_kwargs: Additional backend arguments

    Returns:
        Configured ToneCoach instance
    """
    # Set default models per backend
    if model is None:
        model = {
            "claude": "claude-sonnet-4-20250514",
            "ollama": "qwen2.5:7b",
            "transformers": "Qwen/Qwen2.5-7B-Instruct",
        }.get(backend, "qwen2.5:7b")

    # Configure backend
    backend_kwargs["model"] = model

    # Create LLM backend
    llm = create_backend(backend, **backend_kwargs)

    # Create tool executor
    tools = ToolExecutor(tts=tts, evaluator=evaluator, lazy_load=True)

    # Use local prompt for non-Claude backends
    use_local_prompt = (backend != "claude")

    return ToneCoach(
        llm_backend=llm,
        tool_executor=tools,
        use_local_prompt=use_local_prompt,
    )


def run_interactive(coach: ToneCoach):
    """Run interactive terminal session."""
    print("=" * 60)
    print("  Mandarin Tone Coach")
    print("=" * 60)
    print()
    print("Ask me anything! Try:")
    print("  - 'How do I say hello?'")
    print("  - 'Teach me to count to five'")
    print("  - 'How do I say I love you?'")
    print()
    print("Type 'quit' to exit, 'reset' to start over.")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见! (Goodbye!)")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("再见! (Goodbye!)")
            break

        if user_input.lower() == 'reset':
            coach.reset()
            print("Conversation reset. What would you like to learn?")
            continue

        response = coach.chat(user_input)
        print(f"\nCoach: {response}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mandarin Tone Coach")
    parser.add_argument(
        "--backend",
        choices=["claude", "ollama", "transformers"],
        default="ollama",
        help="LLM backend to use (default: ollama)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (backend-specific)"
    )
    args = parser.parse_args()

    coach = create_coach(backend=args.backend, model=args.model)
    run_interactive(coach)
