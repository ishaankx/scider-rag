"""
Graph traversal tool.
Allows agents to explore entity relationships in the knowledge graph.
"""

from src.agents.tools.base import BaseTool, ToolResult
from src.storage.graph_store import GraphStore


class GraphTraversalTool(BaseTool):
    """Explores entity relationships via multi-hop graph traversal."""

    def __init__(self, graph_store: GraphStore):
        self._graph_store = graph_store

    @property
    def name(self) -> str:
        return "graph_traverse"

    @property
    def description(self) -> str:
        return (
            "Explore relationships between scientific entities in the knowledge graph. "
            "Given an entity name, find all related entities within N hops. "
            "Useful for any scientific domain: researchers, institutions, ML models, "
            "algorithms, genes, compounds, materials, quantum systems, datasets, etc."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": "Name of the entity to start traversal from.",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum number of hops to traverse (1-3, default 2).",
                },
                "relation_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by relationship types (optional).",
                },
            },
            "required": ["entity_name"],
        }

    async def execute(self, **params) -> ToolResult:
        entity_name = params.get("entity_name", "").strip()
        max_depth = min(params.get("max_depth", 2), 3)  # Cap at 3 to prevent expensive queries
        relation_types = params.get("relation_types")

        if not entity_name:
            return ToolResult(output="", success=False, error="Entity name is required.")

        try:
            # First check if entity exists
            entity = await self._graph_store.find_entity(entity_name)
            if not entity:
                return ToolResult(
                    output=f"Entity '{entity_name}' not found in the knowledge graph."
                )

            # Get direct relationships
            direct = await self._graph_store.get_entity_relationships(entity.id)

            # Get multi-hop traversal
            traversal = await self._graph_store.traverse(
                entity_name=entity_name,
                max_depth=max_depth,
                relation_types=relation_types,
            )

            # Format output
            parts = [f"Entity: {entity.name} (type: {entity.entity_type})"]

            if direct:
                parts.append("\nDirect relationships:")
                for rel in direct:
                    arrow = "→" if rel["direction"] == "outgoing" else "←"
                    parts.append(
                        f"  {arrow} {rel['relation_type']} {rel['related_entity']} "
                        f"({rel['related_type']})"
                    )

            if traversal:
                parts.append(f"\nReachable entities (up to {max_depth} hops):")
                for t in traversal:
                    parts.append(
                        f"  [depth {t['depth']}] {t['entity_name']} "
                        f"({t['entity_type']}) via {t['relation_type']}"
                    )

            if not direct and not traversal:
                parts.append("No relationships found for this entity.")

            return ToolResult(output="\n".join(parts))

        except Exception as exc:
            return ToolResult(
                output="Graph traversal failed.",
                success=False,
                error=str(exc),
            )
