"""Validate artifacts against pydantic schemas before registry update."""

from __future__ import annotations

from typing import Type
from pydantic import BaseModel, ValidationError

from ..utils.errors import SchemaValidationError


class SchemaValidationService:
    def validate(self, payload: dict, model: Type[BaseModel]) -> BaseModel:
        try:
            return model.model_validate(payload)
        except ValidationError as e:
            raise SchemaValidationError(
                f"Artifact does not match {model.__name__}", detail={"errors": e.errors()}
            ) from e
