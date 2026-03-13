"""
Entity relationship graph stored in PostgreSQL.
Supports adding entities/relationships and multi-hop traversal
using recursive CTEs — no separate graph database needed.
"""

import uuid

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.models import Entity, Relationship


class GraphStore:
    """Manages the entity-relationship graph in PostgreSQL."""

    def __init__(self, db: AsyncSession):
        self._db = db

    async def upsert_entity(
        self,
        name: str,
        entity_type: str,
        properties: dict | None = None,
        source_document_id: uuid.UUID | None = None,
    ) -> Entity:
        """Insert or return existing entity (matched by normalized name + type)."""
        normalized = name.strip().lower()

        result = await self._db.execute(
            select(Entity).where(
                Entity.normalized_name == normalized,
                Entity.entity_type == entity_type,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            # Merge new properties into existing entity
            if properties:
                merged = {**(existing.properties or {}), **properties}
                existing.properties = merged
                await self._db.flush()
            return existing

        entity = Entity(
            id=uuid.uuid4(),
            name=name.strip(),
            entity_type=entity_type,
            normalized_name=normalized,
            properties=properties or {},
            source_document_id=source_document_id,
        )
        self._db.add(entity)
        await self._db.flush()
        return entity

    async def add_relationship(
        self,
        source_entity_id: uuid.UUID,
        target_entity_id: uuid.UUID,
        relation_type: str,
        weight: float = 1.0,
        properties: dict | None = None,
    ) -> Relationship:
        """Create a directed relationship between two entities."""
        rel = Relationship(
            id=uuid.uuid4(),
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relation_type=relation_type,
            weight=weight,
            properties=properties or {},
        )
        self._db.add(rel)
        await self._db.flush()
        return rel

    async def traverse(
        self,
        entity_name: str,
        max_depth: int = 2,
        relation_types: list[str] | None = None,
    ) -> list[dict]:
        """
        Multi-hop BFS traversal from a starting entity using recursive CTE.
        Returns all reachable entities within max_depth hops.
        """
        normalized = entity_name.strip().lower()

        # Build optional relation type filter
        type_filter = ""
        params: dict = {"name": normalized, "max_depth": max_depth}
        if relation_types:
            type_filter = "AND r.relation_type = ANY(:rel_types)"
            params["rel_types"] = relation_types

        sql = text(f"""
            WITH RECURSIVE graph_walk AS (
                -- Base case: direct relationships from starting entity
                SELECT
                    r.target_entity_id AS entity_id,
                    r.relation_type,
                    e_start.name AS source_name,
                    1 AS depth,
                    ARRAY[e_start.id, r.target_entity_id] AS path
                FROM entities e_start
                JOIN relationships r ON r.source_entity_id = e_start.id
                WHERE e_start.normalized_name = :name
                {type_filter}

                UNION ALL

                -- Recursive step: follow outgoing edges
                SELECT
                    r.target_entity_id,
                    r.relation_type,
                    e.name,
                    gw.depth + 1,
                    gw.path || r.target_entity_id
                FROM graph_walk gw
                JOIN relationships r ON r.source_entity_id = gw.entity_id
                JOIN entities e ON e.id = gw.entity_id
                WHERE gw.depth < :max_depth
                AND NOT (r.target_entity_id = ANY(gw.path))  -- prevent cycles
                {type_filter}
            )
            SELECT DISTINCT
                gw.entity_id,
                gw.relation_type,
                gw.depth,
                e.name AS entity_name,
                e.entity_type,
                e.properties
            FROM graph_walk gw
            JOIN entities e ON e.id = gw.entity_id
            ORDER BY gw.depth, e.name
            LIMIT 200
        """)

        result = await self._db.execute(sql, params)
        rows = result.mappings().all()

        return [
            {
                "entity_id": str(row["entity_id"]),
                "entity_name": row["entity_name"],
                "entity_type": row["entity_type"],
                "relation_type": row["relation_type"],
                "depth": row["depth"],
                "properties": row["properties"],
            }
            for row in rows
        ]

    async def find_entity(self, name: str) -> Entity | None:
        """Look up an entity by normalized name."""
        normalized = name.strip().lower()
        result = await self._db.execute(
            select(Entity).where(Entity.normalized_name == normalized)
        )
        return result.scalar_one_or_none()

    async def get_entity_relationships(self, entity_id: uuid.UUID) -> list[dict]:
        """Get all direct relationships (both incoming and outgoing) for an entity."""
        sql = text("""
            SELECT
                r.relation_type,
                'outgoing' AS direction,
                e.name AS related_entity,
                e.entity_type AS related_type
            FROM relationships r
            JOIN entities e ON e.id = r.target_entity_id
            WHERE r.source_entity_id = :eid

            UNION ALL

            SELECT
                r.relation_type,
                'incoming' AS direction,
                e.name AS related_entity,
                e.entity_type AS related_type
            FROM relationships r
            JOIN entities e ON e.id = r.source_entity_id
            WHERE r.target_entity_id = :eid
        """)

        result = await self._db.execute(sql, {"eid": entity_id})
        rows = result.mappings().all()

        return [dict(row) for row in rows]
