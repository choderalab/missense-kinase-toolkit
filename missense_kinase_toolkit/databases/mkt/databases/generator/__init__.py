"""Compositional build pipeline for assembling and serializing KinaseInfo objects.

Splits the ``generate_kinaseinfo_objects`` workflow into a base build (API fetch +
harmonization), a registry of idempotent enrichment steps that mutate additive optional
fields, and a thin orchestration layer supporting full regeneration, per-step
(``--only``/``--skip``), and per-entry (``--kinase``) run modes.
"""
