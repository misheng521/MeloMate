# Prompts

This directory contains utility prompts used in MeloMate. These are general-purpose prompts that are not specific to any character's persona.

## Examples of Utility Prompts

*   **Avatar Expressions:** Prompts that inform the LLM about available avatar expressions.
*   **Tool Usage:** Prompts that guide the LLM on how to use available tools.
*   ... and many more.

## Character Persona Prompts

**Important:** Character persona texts describe a character's background and tendencies, rather than a checklist to perform in every reply. They are **NOT** stored in this directory.

They are located in:
* `characters/profiles/*.md` or `*.txt`, relative to the repository root.
* Existing YAML profiles remain supported; a same-name text file overrides only
  their persona prompt, preserving character IDs and technical configuration.

A text prompt alone creates a new selectable character. The application rereads
the selected text prompt on each conversation turn. No model writes this file.
Generated notes live separately in `characters/memory/<conf_uid>/memory.md`.
The shared interpretation in `persona_text.py` distinguishes explicit background
facts from contextual tendencies. Conversation, proactive opportunities and memory
review preserve this distinction; repeated persona descriptions do not become
independent evidence of a permanent trait. User-authored texts are not rewritten.
See [人设与记忆说明](../../docs/MEMORY.md) for editing and migration behavior.
