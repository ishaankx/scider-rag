"""
SQLAlchemy ORM models for the application database.

Tables:
- documents: Ingested source files (PDFs, CSVs, etc.)
- chunks: Text chunks extracted from documents, linked to vector embeddings.
- entities: Named entities extracted from documents (authors, models, genes, algorithms, materials, qubits, etc.)
- relationships: Directed edges between entities (e.g., "authored_by", "outperforms", "uses", "interacts_with").
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    source_type = Column(String(50), nullable=False)  # "pdf", "csv", "json"
    file_name = Column(String(500), nullable=False)
    content_hash = Column(String(64), nullable=False, unique=True)  # SHA-256
    metadata_ = Column("metadata", JSONB, default=dict)
    chunk_count = Column(Integer, default=0)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    entities = relationship("Entity", back_populates="source_document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_documents_source_type", "source_type"),
        Index("ix_documents_content_hash", "content_hash"),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    token_count = Column(Integer, default=0)
    embedding_id = Column(String(100), nullable=True)  # ID in Qdrant
    metadata_ = Column("metadata", JSONB, default=dict)  # section, page_number, etc.
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
    )


class Entity(Base):
    __tablename__ = "entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(500), nullable=False)
    entity_type = Column(String(100), nullable=False)  # "person", "gene", "compound", etc.
    normalized_name = Column(String(500), nullable=False)  # lowercase, stripped
    properties = Column(JSONB, default=dict)
    source_document_id = Column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=True
    )
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    source_document = relationship("Document", back_populates="entities")

    # Relationships where this entity is the source
    outgoing_relationships = relationship(
        "Relationship",
        foreign_keys="Relationship.source_entity_id",
        back_populates="source_entity",
        cascade="all, delete-orphan",
    )
    # Relationships where this entity is the target
    incoming_relationships = relationship(
        "Relationship",
        foreign_keys="Relationship.target_entity_id",
        back_populates="target_entity",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_entities_normalized_name", "normalized_name"),
        Index("ix_entities_type", "entity_type"),
        Index("ix_entities_name_type", "normalized_name", "entity_type", unique=True),
    )


class Relationship(Base):
    __tablename__ = "relationships"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_entity_id = Column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    target_entity_id = Column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    relation_type = Column(String(200), nullable=False)  # "authored_by", "inhibits", etc.
    weight = Column(Float, default=1.0)
    properties = Column(JSONB, default=dict)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    source_entity = relationship(
        "Entity", foreign_keys=[source_entity_id], back_populates="outgoing_relationships"
    )
    target_entity = relationship(
        "Entity", foreign_keys=[target_entity_id], back_populates="incoming_relationships"
    )

    __table_args__ = (
        Index("ix_relationships_source", "source_entity_id"),
        Index("ix_relationships_target", "target_entity_id"),
        Index("ix_relationships_type", "relation_type"),
    )
