"""Structure handling: parse PDB/CIF, extract chain ids, normalize structure refs.

Also owns the RFdiffusion `contigs_dsl` template/validator. The
StructureAndDesignAgent is NOT allowed to write contigs_dsl directly — it
selects mode/parameters and this service emits the validated DSL string.
"""

from __future__ import annotations


class StructureService:
    def parse_structure(self, path: str) -> dict:
        raise NotImplementedError

    def extract_chain_ids(self, parsed: dict) -> list[str]:
        raise NotImplementedError

    def build_rfdiffusion_contigs_dsl(self, params: dict) -> str:
        raise NotImplementedError

    def validate_contigs_dsl(self, dsl: str) -> None:
        raise NotImplementedError
