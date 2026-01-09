"""
Prompt Template Management System
=================================
Centralized prompt management using Jinja2 templates.

Features:
- External template files (YAML/Jinja2)
- Variable injection with validation
- Version tracking for prompts
- Squad-specific prompt loading
- RAG context integration

Design Principles:
- Prompts are data, not code - externalize them
- Clear separation between system prompts and user prompts
- Consistent formatting for structured outputs
- Easy A/B testing and iteration
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Base path for prompt templates
PROMPTS_DIR = Path(__file__).parent
TEMPLATES_DIR = PROMPTS_DIR / "templates"


# =============================================================================
# PROMPT TEMPLATE MODELS
# =============================================================================

class PromptMetadata(BaseModel):
    """Metadata for a prompt template."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    agent: str = ""
    squad: str = ""
    author: str = "system"
    tags: list[str] = Field(default_factory=list)


class PromptSection(BaseModel):
    """A section of a prompt (role, context, task, etc.)."""
    name: str
    content: str
    required: bool = True


class PromptTemplate(BaseModel):
    """Complete prompt template with metadata and sections."""
    metadata: PromptMetadata
    system_prompt: str
    user_prompt_template: str = "{input}"
    sections: dict[str, PromptSection] = Field(default_factory=dict)
    variables: list[str] = Field(default_factory=list)
    output_format: str | None = None


# =============================================================================
# PROMPT MANAGER
# =============================================================================

class PromptManager:
    """
    Centralized prompt template manager.

    Loads, caches, and renders prompt templates from external files.

    Usage:
        >>> manager = PromptManager()
        >>> prompt = manager.get_prompt("meeting_notes")
        >>> rendered = manager.render("meeting_notes", transcript="...")
    """

    _instance: PromptManager | None = None
    _templates: dict[str, PromptTemplate] = {}

    def __init__(self, templates_dir: Path | None = None):
        """
        Initialize the prompt manager.

        Args:
            templates_dir: Path to templates directory (default: prompts/templates)
        """
        self.templates_dir = templates_dir or TEMPLATES_DIR

        # Ensure templates directory exists
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Add custom filters
        self._register_filters()

        # Load all templates
        self._load_templates()

        logger.info(f"PromptManager initialized with {len(self._templates)} templates")

    @classmethod
    def get_instance(cls) -> PromptManager:
        """Get singleton instance of PromptManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _register_filters(self) -> None:
        """Register custom Jinja2 filters."""
        # Truncate text to max length
        self.jinja_env.filters["truncate_smart"] = lambda s, length: (
            s[:length-3] + "..." if len(s) > length else s
        )

        # Format list as bullet points
        self.jinja_env.filters["as_bullets"] = lambda items: "\n".join(
            f"- {item}" for item in items
        )

        # Format dict as key-value pairs
        self.jinja_env.filters["as_pairs"] = lambda d: "\n".join(
            f"- {k}: {v}" for k, v in d.items()
        )

    def _load_templates(self) -> None:
        """Load all YAML templates from templates directory."""
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return

        for yaml_file in self.templates_dir.glob("*.yaml"):
            try:
                self._load_template_file(yaml_file)
            except Exception as e:
                logger.error(f"Failed to load template {yaml_file}: {e}")

    def _load_template_file(self, file_path: Path) -> None:
        """Load a single template file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return

        # Handle single template or multiple templates in one file
        templates = data if isinstance(data, list) else [data]

        for template_data in templates:
            try:
                metadata = PromptMetadata(**template_data.get("metadata", {}))

                prompt = PromptTemplate(
                    metadata=metadata,
                    system_prompt=template_data.get("system_prompt", ""),
                    user_prompt_template=template_data.get("user_prompt_template", "{input}"),
                    variables=template_data.get("variables", []),
                    output_format=template_data.get("output_format"),
                )

                # Load sections if present
                if "sections" in template_data:
                    for section_name, section_data in template_data["sections"].items():
                        prompt.sections[section_name] = PromptSection(
                            name=section_name,
                            **section_data if isinstance(section_data, dict) else {"content": section_data}
                        )

                self._templates[metadata.name] = prompt
                logger.debug(f"Loaded template: {metadata.name} v{metadata.version}")

            except Exception as e:
                logger.error(f"Failed to parse template in {file_path}: {e}")

    def get_prompt(self, name: str) -> PromptTemplate | None:
        """
        Get a prompt template by name.

        Args:
            name: Template name

        Returns:
            PromptTemplate or None if not found
        """
        return self._templates.get(name)

    def render(
        self,
        name: str,
        **variables: Any,
    ) -> str:
        """
        Render a prompt template with variables.

        Args:
            name: Template name
            **variables: Variables to inject into template

        Returns:
            Rendered prompt string

        Raises:
            KeyError: If template not found
        """
        template = self.get_prompt(name)
        if not template:
            raise KeyError(f"Prompt template not found: {name}")

        # Render system prompt with Jinja2
        jinja_template = Template(template.system_prompt)
        return jinja_template.render(**variables)

    def render_user_prompt(
        self,
        name: str,
        **variables: Any,
    ) -> str:
        """
        Render the user prompt template.

        Args:
            name: Template name
            **variables: Variables to inject

        Returns:
            Rendered user prompt string
        """
        template = self.get_prompt(name)
        if not template:
            raise KeyError(f"Prompt template not found: {name}")

        jinja_template = Template(template.user_prompt_template)
        return jinja_template.render(**variables)

    def get_full_prompt(
        self,
        name: str,
        include_format: bool = True,
        format_instructions: str | None = None,
        **variables: Any,
    ) -> tuple[str, str]:
        """
        Get both system and user prompts rendered.

        Args:
            name: Template name
            include_format: Whether to include output format section
            format_instructions: Pydantic format instructions to include
            **variables: Variables for rendering

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        template = self.get_prompt(name)
        if not template:
            raise KeyError(f"Prompt template not found: {name}")

        # Render system prompt
        system = self.render(name, **variables)

        # Add output format if requested
        if include_format and format_instructions:
            system += f"\n\n## Output Format\n{format_instructions}"
        elif include_format and template.output_format:
            system += f"\n\n## Output Format\n{template.output_format}"

        # Render user prompt
        user = self.render_user_prompt(name, **variables)

        return system, user

    def list_templates(self, squad: str | None = None) -> list[str]:
        """
        List available template names.

        Args:
            squad: Optional filter by squad

        Returns:
            List of template names
        """
        if squad:
            return [
                name for name, template in self._templates.items()
                if template.metadata.squad == squad
            ]
        return list(self._templates.keys())

    def register_template(self, template: PromptTemplate) -> None:
        """
        Register a template programmatically.

        Args:
            template: PromptTemplate to register
        """
        self._templates[template.metadata.name] = template
        logger.info(f"Registered template: {template.metadata.name}")


# =============================================================================
# PROMPT BUILDER (Fluent API)
# =============================================================================

class PromptBuilder:
    """
    Fluent builder for constructing prompts programmatically.

    Usage:
        >>> prompt = (PromptBuilder("my_agent")
        ...     .set_role("You are a helpful assistant")
        ...     .add_context("Client: {client_name}")
        ...     .add_task("Analyze the following: {input}")
        ...     .set_output_format("JSON with 'result' key")
        ...     .build())
    """

    def __init__(self, name: str):
        self.name = name
        self._role: str = ""
        self._context: list[str] = []
        self._constraints: list[str] = []
        self._task: str = ""
        self._examples: list[dict[str, str]] = []
        self._output_format: str = ""
        self._metadata: dict[str, Any] = {}

    def set_role(self, role: str) -> PromptBuilder:
        """Set the agent's role description."""
        self._role = role
        return self

    def add_context(self, context: str) -> PromptBuilder:
        """Add context information."""
        self._context.append(context)
        return self

    def add_constraint(self, constraint: str) -> PromptBuilder:
        """Add a behavioral constraint."""
        self._constraints.append(constraint)
        return self

    def set_task(self, task: str) -> PromptBuilder:
        """Set the main task description."""
        self._task = task
        return self

    def add_example(self, input_text: str, output_text: str) -> PromptBuilder:
        """Add an input/output example."""
        self._examples.append({"input": input_text, "output": output_text})
        return self

    def set_output_format(self, format_desc: str) -> PromptBuilder:
        """Set output format description."""
        self._output_format = format_desc
        return self

    def set_metadata(self, **kwargs: Any) -> PromptBuilder:
        """Set metadata fields."""
        self._metadata.update(kwargs)
        return self

    def build(self) -> PromptTemplate:
        """Build the final PromptTemplate."""
        # Construct system prompt
        parts = []

        if self._role:
            parts.append(f"# Role\n{self._role}")

        if self._context:
            parts.append(f"# Context\n" + "\n".join(self._context))

        if self._constraints:
            parts.append(f"# Constraints\n" + "\n".join(f"- {c}" for c in self._constraints))

        if self._task:
            parts.append(f"# Task\n{self._task}")

        if self._examples:
            examples_text = "\n\n".join(
                f"Input: {ex['input']}\nOutput: {ex['output']}"
                for ex in self._examples
            )
            parts.append(f"# Examples\n{examples_text}")

        system_prompt = "\n\n".join(parts)

        return PromptTemplate(
            metadata=PromptMetadata(
                name=self.name,
                **self._metadata,
            ),
            system_prompt=system_prompt,
            output_format=self._output_format,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_prompt(name: str) -> PromptTemplate | None:
    """Get a prompt template by name (convenience function)."""
    return PromptManager.get_instance().get_prompt(name)


def render_prompt(name: str, **variables: Any) -> str:
    """Render a prompt template (convenience function)."""
    return PromptManager.get_instance().render(name, **variables)


def build_prompt(name: str) -> PromptBuilder:
    """Create a new prompt builder (convenience function)."""
    return PromptBuilder(name)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Models
    "PromptMetadata",
    "PromptSection",
    "PromptTemplate",
    # Manager
    "PromptManager",
    # Builder
    "PromptBuilder",
    # Convenience functions
    "get_prompt",
    "render_prompt",
    "build_prompt",
    # Constants
    "PROMPTS_DIR",
    "TEMPLATES_DIR",
]
